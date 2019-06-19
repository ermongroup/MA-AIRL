import os.path as osp
import time

import joblib
import numpy as np
import tensorflow as tf
from rl.acktr.utils import Scheduler, find_trainable_variables, discount_with_dones
from rl.acktr.utils import cat_entropy, mse
from rl import logger
from rl.acktr import kfac
from rl.common import set_global_seeds, explained_variance


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=2, nsteps=200,
                 nstack=1, ent_coef=0.00, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear', identical=None, use_kfac=True):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        nbatch = nenvs * nsteps
        ob_space = [ob_space]
        ac_space = [ac_space]
        self.num_agents = num_agents = len(ob_space)
        if identical is None:
            identical = [False for _ in range(self.num_agents)]

        scale = [1 for _ in range(num_agents)]
        pointer = [i for i in range(num_agents)]
        h = 0
        for k in range(num_agents):
            if identical[k]:
                scale[h] += 1
            else:
                pointer[h] = k
                h = k
        pointer[h] = num_agents

        print(pointer)

        A, ADV, R, PG_LR = [], [], [], []
        for k in range(num_agents):
            if identical[k]:
                A.append(A[-1])
                ADV.append(ADV[-1])
                R.append(R[-1])
                PG_LR.append(PG_LR[-1])
            else:
                A.append(tf.placeholder(tf.int32, [nbatch * scale[k], ac_space[k].shape[0]]))
                ADV.append(tf.placeholder(tf.float32, [nbatch * scale[k]]))
                R.append(tf.placeholder(tf.float32, [nbatch * scale[k]]))
                PG_LR.append(tf.placeholder(tf.float32, []))

        # A = [tf.placeholder(tf.int32, [nbatch]) for _ in range(num_agents)]
        # ADV = [tf.placeholder(tf.float32, [nbatch]) for _ in range(num_agents)]
        # R = [tf.placeholder(tf.float32, [nbatch]) for _ in range(num_agents)]
        # PG_LR = [tf.placeholder(tf.float32, []) for _ in range(num_agents)]
        # VF_LR = [tf.placeholder(tf.float32, []) for _ in range(num_agents)]
        pg_loss, entropy, vf_loss, train_loss = [], [], [], []
        self.model = step_model = []
        self.model2 = train_model = []
        self.pg_fisher = pg_fisher_loss = []
        self.logits = logits = []
        sample_net = []
        self.vf_fisher = vf_fisher_loss = []
        self.joint_fisher = joint_fisher_loss = []
        self.lld = lld = []

        for k in range(num_agents):
            if identical[k]:
                step_model.append(step_model[-1])
                train_model.append(train_model[-1])
            else:
                step_model.append(policy(sess, ob_space[k], ac_space[k], ob_space, ac_space,
                                         nenvs, 1, nstack, reuse=False, name='%d' % k))
                train_model.append(policy(sess, ob_space[k], ac_space[k], ob_space, ac_space,
                                          nenvs * scale[k], nsteps, nstack, reuse=True, name='%d' % k))

            ac = ac_space[k].shape[0]
            logpac = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=train_model[k].pi, labels=A[k]), axis=1)
            # logpac = 0.5 * tf.reduce_sum(tf.square((A[k] - stats[:, :ac]) / stats[:, ac:]), axis=-1) \
            #          + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(A[k])[-1]) \
            #          + tf.reduce_sum(tf.log(stats[:, ac:]), axis=-1)
            lld.append(tf.reduce_mean(logpac))

            ##training loss
            pg_loss.append(tf.reduce_mean(ADV[k] * logpac))
            entropy.append(tf.reduce_mean(cat_entropy(train_model[k].pi)))
            # entropy.append(tf.reduce_sum(
            #     train_model[k].logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1))
            pg_loss[k] = pg_loss[k] - ent_coef * entropy[k]
            vf_loss.append(tf.reduce_mean(mse(tf.squeeze(train_model[k].vf), R[k])))
            train_loss.append(pg_loss[k] + vf_coef * vf_loss[k])

            ##Fisher loss construction
            pg_fisher_loss.append(-tf.reduce_mean(logpac))
            sample_net.append(train_model[k].vf + tf.random_normal(tf.shape(train_model[k].vf)))
            vf_fisher_loss.append(-vf_fisher_coef * tf.reduce_mean(
                tf.pow(train_model[k].vf - tf.stop_gradient(sample_net[k]), 2)))
            joint_fisher_loss.append(pg_fisher_loss[k] + vf_fisher_loss[k])

        self.policy_params = [] # [find_trainable_variables("policy_%d" % k) for k in range(num_agents)]
        self.value_params = [] # [find_trainable_variables('value_%d' % k) for k in range(num_agents)]

        for k in range(num_agents):
            if identical[k]:
                self.policy_params.append(self.policy_params[-1])
                self.value_params.append(self.value_params[-1])
            else:
                self.policy_params.append(find_trainable_variables("policy_%d" % k))
                self.value_params.append(find_trainable_variables("value_%d" % k))

        self.params = params = [a + b for a, b in zip(self.policy_params, self.value_params)]
        params_flat = []
        for k in range(num_agents):
            params_flat.extend(params[k])

        self.grads_check = grads = [
            tf.gradients(train_loss[k], params[k]) for k in range(num_agents)
        ]
        clone_grads = [
            tf.gradients(lld[k], params[k]) for k in range(num_agents)
        ]

        self.optim = optim = []
        self.clones = clones = []
        update_stats_op = []
        train_op, clone_op, q_runner = [], [], []

        if use_kfac:
            for k in range(num_agents):
                if identical[k]:
                    optim.append(optim[-1])
                    train_op.append(train_op[-1])
                    q_runner.append(q_runner[-1])
                    clones.append(clones[-1])
                    clone_op.append(clone_op[-1])
                else:
                    with tf.variable_scope('optim_%d' % k):
                        optim.append(kfac.KfacOptimizer(
                            learning_rate=PG_LR[k], clip_kl=kfac_clip,
                            momentum=0.9, kfac_update=1, epsilon=0.01,
                            stats_decay=0.99, async=0, cold_iter=10,
                            max_grad_norm=max_grad_norm)
                        )
                        update_stats_op.append(optim[k].compute_and_apply_stats(joint_fisher_loss[k], var_list=params[k]))
                        train_op_, q_runner_ = optim[k].apply_gradients(list(zip(grads[k], params[k])))
                        train_op.append(train_op_)
                        q_runner.append(q_runner_)

                    with tf.variable_scope('clone_%d' % k):
                        clones.append(kfac.KfacOptimizer(
                            learning_rate=PG_LR[k], clip_kl=kfac_clip,
                            momentum=0.9, kfac_update=1, epsilon=0.01,
                            stats_decay=0.99, async=0, cold_iter=10,
                            max_grad_norm=max_grad_norm)
                        )
                        update_stats_op.append(clones[k].compute_and_apply_stats(
                            pg_fisher_loss[k], var_list=self.policy_params[k]))
                        clone_op_, q_runner_ = clones[k].apply_gradients(list(zip(clone_grads[k], self.policy_params[k])))
                        clone_op.append(clone_op_)

        update_stats_op = tf.group(*update_stats_op)
        train_ops = train_op
        # train_op = tf.group(*train_op)
        clone_ops = clone_op
        # clone_op = tf.group(*clone_op)

        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        self.clone_lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = [rewards[k] - values[k] for k in range(num_agents)]
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            ob = np.concatenate(obs, axis=1)
            int_actions = [np.minimum(10.0, np.maximum(0.0, np.floor((ac + 1.1) / 0.2))) for ac in actions]
            int_actions = [ac.astype(np.int) for ac in int_actions]

            td_map = {}
            for k in range(num_agents):
                if identical[k]:
                    continue
                new_map = {}
                if num_agents > 1:
                    action_v = []
                    for j in range(k, pointer[k]):
                        action_v.append(np.concatenate([actions[i] for i in range(num_agents) if i != k], axis=1))
                    action_v = np.concatenate(action_v, axis=0)
                    new_map.update({train_model[k].A_v: action_v})
                    td_map.update({train_model[k].A_v: action_v})

                new_map.update({
                    train_model[k].X: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    train_model[k].X_v: np.concatenate([ob.copy() for j in range(k, pointer[k])], axis=0),
                    A[k]: np.concatenate([int_actions[j] for j in range(k, pointer[k])], axis=0),
                    ADV[k]: np.concatenate([advs[j] for j in range(k, pointer[k])], axis=0),
                    R[k]: np.concatenate([rewards[j] for j in range(k, pointer[k])], axis=0),
                    PG_LR[k]: cur_lr / float(scale[k])
                })
                sess.run(train_ops[k], feed_dict=new_map)
                td_map.update(new_map)

                if states[k] != []:
                    td_map[train_model[k].S] = states
                    td_map[train_model[k].M] = masks

            policy_loss, value_loss, policy_entropy = sess.run(
                [pg_loss, vf_loss, entropy],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def clone(obs, actions):
            td_map = {}
            cur_lr = self.clone_lr.value()
            int_actions = [np.minimum(10.0, np.maximum(0.0, np.floor((ac + 1.1) / 0.2))) for ac in actions]
            int_actions = [ac.astype(np.int) for ac in int_actions]
            for k in range(num_agents):
                new_map = {
                    train_model[k].X: obs[k],
                    A[k]: int_actions[k],
                    PG_LR[k]: cur_lr
                }
                td_map.update(new_map)
                sess.run(clone_op, new_map)
            lld_loss = sess.run([lld], td_map)
            return lld_loss

        def save(save_path):
            ps = sess.run(params_flat)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params_flat, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.clone = clone
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model

        def step(ob, av, *_args, **_kwargs):
            a, v, s = [], [], []
            obs = np.concatenate(ob, axis=1)
            for k in range(num_agents):
                if num_agents > 1:
                    a_v = np.concatenate([av[i] for i in range(num_agents) if i != k], axis=1)
                else:
                    a_v = None
                a_, v_, s_ = step_model[k].step(ob[k], obs, a_v)
                a.append(a_)
                v.append(v_)
                s.append(s_)
            return a, v, s

        self.step = step

        def value(obs, av):
            v = []
            ob = np.concatenate(obs, axis=1)
            for k in range(num_agents):
                if num_agents > 1:
                    a_v = np.concatenate([av[i] for i in range(num_agents) if i != k], axis=1)
                else:
                    a_v = None
                v_ = step_model[k].value(ob, a_v)
                v.append(v_)
            return v

        self.value = value
        self.initial_state = [step_model[k].initial_state for k in range(num_agents)]
        tf.global_variables_initializer().run(session=sess)


class Runner(object):
    def __init__(self, env, model, nsteps, nstack, gamma, lam):
        self.env = env
        self.model = model
        ob_space = [env.observation_space]
        ac_space = [env.action_space]
        self.num_agents = len(ob_space)
        self.nenv = nenv = env.num_envs
        self.batch_ob_shape = [
            (nenv * nsteps, nstack * ob_space[k].shape[0]) for k in range(self.num_agents)]
        self.obs = [
            np.zeros((nenv, nstack * ob_space[k].shape[0])) for k in range(self.num_agents)
        ]
        self.actions = [
            np.zeros((nenv, )) for k in range(self.num_agents)
        ]
        obs = env.reset()
        obs = [obs]
        self.update_obs(obs)
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [np.array([False for _ in range(nenv)]) for k in range(self.num_agents)]

    def update_obs(self, obs):
        self.obs = obs
        # for k in range(self.num_agents):
        #     ob = np.roll(self.obs[k], shift=-1, axis=1)
        #     ob[:, -1] = obs[:, 0]
        #     self.obs[k] = ob

        # self.obs = [np.roll(ob, shift=-1, axis=3) for ob in self.obs]
        # self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def run(self):
        mb_obs = [[] for _ in range(self.num_agents)]
        mb_rewards = [[] for _ in range(self.num_agents)]
        mb_actions = [[] for _ in range(self.num_agents)]
        mb_values = [[] for _ in range(self.num_agents)]
        mb_dones = [[] for _ in range(self.num_agents)]
        mb_masks = [[] for _ in range(self.num_agents)]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.actions)
            self.actions = actions
            for k in range(self.num_agents):
                mb_obs[k].append(np.copy(self.obs[k]))
                mb_actions[k].append(actions[k])
                mb_values[k].append(values[k])
                mb_dones[k].append(self.dones[k])
            actions_list = []
            for i in range(self.nenv):
                actions_list.append([actions[k][i] for k in range(self.num_agents)])
            obs, rewards, dones, _ = self.env.step(actions_list[0])
            obs, rewards, dones = [obs], [rewards], [dones]
            self.states = states
            self.dones = dones
            for k in range(self.num_agents):
                for ni, done in enumerate(dones[k]):
                    if done:
                        self.obs[k][ni] = self.obs[k][ni] * 0.0
            self.update_obs(obs)
            for k in range(self.num_agents):
                mb_rewards[k].append(rewards[k])
            # mb_rewards.append(rewards)
        for k in range(self.num_agents):
            mb_dones[k].append(self.dones[k])

        # batch of steps to batch of rollouts
        for k in range(self.num_agents):
            mb_obs[k] = np.asarray(mb_obs[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])
            mb_rewards[k] = np.asarray(mb_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_actions[k] = np.asarray(mb_actions[k], dtype=np.int32).swapaxes(1, 0)
            mb_values[k] = np.asarray(mb_values[k], dtype=np.float32).swapaxes(1, 0)
            mb_dones[k] = np.asarray(mb_dones[k], dtype=np.bool).swapaxes(1, 0)
            mb_masks[k] = mb_dones[k][:, :-1]
            mb_dones[k] = mb_dones[k][:, 1:]

        last_values = self.model.value(self.obs, self.actions) # self.states, self.dones)

        mb_advs = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        mb_returns = [[] for _ in range(self.num_agents)]

        lastgaelam = 0.0
        for k in range(self.num_agents):
            for t in reversed(range(self.nsteps)):
                if t == self.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones[k]
                    nextvalues = last_values[k]
                else:
                    nextnonterminal = 1.0 - mb_dones[k][:, t + 1]
                    nextvalues = mb_values[k][:, t + 1]
                delta = mb_rewards[k][:, t] + self.gamma * nextvalues * nextnonterminal - mb_values[k][:, t]
                mb_advs[k][:, t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            mb_returns[k] = mb_advs[k] + mb_values[k]
            mb_returns[k] = mb_returns[k].flatten()
            mb_masks[k] = mb_masks[k].flatten()
            mb_values[k] = mb_values[k].flatten()
            mb_actions[k] = np.reshape(mb_actions[k], [-1, mb_actions[k].shape[-1]])

        # mb_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        # last_values = self.model.value(self.obs, self.actions)
        # # discount/bootstrap off value fn
        # for k in range(self.num_agents):
        #     for n, (rewards, dones, value) in enumerate(zip(mb_rewards[k], mb_dones[k], last_values[k].tolist())):
        #         rewards = rewards.tolist()
        #         dones = dones.tolist()
        #         if dones[-1] == 0:
        #             rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
        #         else:
        #             rewards = discount_with_dones(rewards, dones, self.gamma)
        #         mb_returns[k][n] = rewards
        #
        # for k in range(self.num_agents):
        #     mb_returns[k] = mb_returns[k].flatten()
        #     mb_masks[k] = mb_masks[k].flatten()
        #     mb_values[k] = mb_values[k].flatten()
        #     mb_actions[k] = np.reshape(mb_actions[k], [-1, mb_actions[k].shape[-1]])

        return mb_obs, mb_states, mb_returns, mb_masks, mb_actions, mb_values


def learn(policy, env, seed, total_timesteps=int(40e6), gamma=0.995, lam=0.95, log_interval=1, nprocs=32, nsteps=20,
          nstack=1, ent_coef=0.00, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=100, lrschedule='linear', identical=None):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    make_model = lambda: Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps
                                =nsteps, nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=
                                vf_fisher_coef, lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                                lrschedule=lrschedule, identical=identical)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()

    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma, lam=lam)
    nbatch = nenvs*nsteps
    tstart = time.time()
    coord = tf.train.Coordinator()
    # enqueue_threads = [q_runner.create_threads(model.sess, coord=coord, start=True) for q_runner in model.q_runner]
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        model.old_obs = obs
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = [explained_variance(values[k], rewards[k]) for k in range(model.num_agents)]
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            for k in range(model.num_agents):
                # logger.record_tabular('reward %d' % k, np.mean(rewards[k]))
                logger.record_tabular("explained_variance %d" % k, float(ev[k]))
                logger.record_tabular("policy_entropy %d" % k, float(policy_entropy[k]))
                logger.record_tabular("policy_loss %d" % k, float(policy_loss[k]))
                logger.record_tabular("value_loss %d" % k, float(value_loss[k]))
            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'checkpoint%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
    coord.request_stop()
    # coord.join(enqueue_threads)
    env.close()


