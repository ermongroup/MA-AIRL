import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from rl import logger
from collections import deque
from baselines.common import explained_variance


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def lsf01(larr):
    l = []
    for arr in larr:
        s = arr.shape
        l.append(arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:]))
    return l


class Model(object):
    def __init__(self, *, policy, ob_spaces, ac_spaces, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        self.num_agents = num_agents = len(ob_spaces)
        act_model, train_model = [], []
        for k in range(num_agents):
            # TODO: values
            # if k > 0:
            #     act_model.append(act_model[-1])
            #     train_model.append(train_model[-1])
            # else:
            act_model.append(policy(sess, ob_spaces, ac_spaces, nbatch_act, 1, reuse=False, idx=k))
            train_model.append(policy(sess, ob_spaces, ac_spaces, nbatch_train, nsteps, reuse=True, idx=k))

        A, ADV, R, OLDNEGLOGPAC, OLDVPRED, CLIPRANGE = [], [], [], [], [], []
        LR = tf.placeholder(tf.float32, [])
        for k in range(num_agents):
            A.append(train_model[k].pdtype.sample_placeholder([None]))
            ADV.append(tf.placeholder(tf.float32, [None]))
            R.append(tf.placeholder(tf.float32, [None]))
            OLDNEGLOGPAC.append(tf.placeholder(tf.float32, [None]))
            OLDVPRED.append(tf.placeholder(tf.float32, [None]))
            CLIPRANGE.append(tf.placeholder(tf.float32, []))

        neglogpac = [train_model[k].pd.neglogp(A[k]) for k in range(num_agents)]
        entropy = [tf.reduce_mean(train_model[k].pd.entropy()) for k in range(num_agents)]

        vpred = [train_model[k].vf for k in range(num_agents)]
        vpredclipped = [OLDVPRED[k] + tf.clip_by_value(train_model[k].vf - OLDVPRED[k], - CLIPRANGE[k], CLIPRANGE[k]) for k in range(num_agents)]
        vf_losses1 = [tf.square(vpred[k] - R[k]) for k in range(num_agents)]
        vf_losses2 = [tf.square(vpredclipped[k] - R[k]) for k in range(num_agents)]
        vf_loss = [.5 * tf.reduce_mean(tf.maximum(vf_losses1[k], vf_losses2[k])) for k in range(num_agents)]
        ratio = [tf.exp(OLDNEGLOGPAC[k] - neglogpac[k]) for k in range(num_agents)]
        pg_losses = [-ADV[k] * ratio[k] for k in range(num_agents)]
        pg_losses2 = [-ADV[k] * tf.clip_by_value(ratio[k], 1.0 - CLIPRANGE[k], 1.0 + CLIPRANGE[k]) for k in range(num_agents)]
        pg_loss = [tf.reduce_mean(tf.maximum(pg_losses[k], pg_losses2[k])) for k in range(num_agents)]
        approxkl = [.5 * tf.reduce_mean(tf.square(neglogpac[k] - OLDNEGLOGPAC[k])) for k in range(num_agents)]
        clipfrac = [tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio[k] - 1.0), CLIPRANGE[k]))) for k in range(num_agents)]
        loss = [pg_loss[k] - entropy[k] * ent_coef + vf_loss[k] * vf_coef for k in range(num_agents)]

        params = []
        for k in range(num_agents):
            with tf.variable_scope('model_{}'.format(k)):
                params.append(tf.trainable_variables())
        grads = [tf.gradients(loss[k], params[k]) for k in range(num_agents)]
        grads_clip, grads_norm = [], []
        if max_grad_norm is not None:
            for k in range(num_agents):
                _grads, _grad_norm = tf.clip_by_global_norm(grads[k], max_grad_norm)
                grads_clip.append(_grads)
                grads_norm.append(_grad_norm)
            grads = grads_clip
        grads = [list(zip(grads[k], params[k])) for k in range(num_agents)]
        grads = sum(grads, [])  # list of list to list
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        clone_grads = [tf.gradients(tf.reduce_mean(neglogpac[k]), params[k]) for k in range(num_agents)]
        grads_clip, grads_norm = [], []
        if max_grad_norm is not None:
            for k in range(num_agents):
                _grads, _grad_norm = tf.clip_by_global_norm(clone_grads[k], max_grad_norm)
                grads_clip.append(_grads)
                grads_norm.append(_grad_norm)
            clone_grads = grads_clip
        clone_grads = [list(zip(clone_grads[k], params[k])) for k in range(num_agents)]
        clone_grads = sum(clone_grads, [])  # list of list to list
        cloner = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _clone = cloner.apply_gradients(clone_grads)

        params_flat = []
        for k in range(num_agents):
            params_flat.extend(params[k])

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, advs=None, states=None):
            advs = [returns[k] - values[k] for k in range(num_agents)]
            advs = [(advs[k] - advs[k].mean()) / (advs[k].std() + 1e-8) for k in range(num_agents)]
            td_map = {}
            for k in range(num_agents):
                new_map = {train_model[k].X:obs[k], A[k]:actions[k], ADV[k]:advs[k], R[k]:returns[k], LR:lr,
                        CLIPRANGE[k]:cliprange, OLDNEGLOGPAC[k]:neglogpacs[k], OLDVPRED[k]:values[k],
                           train_model[k].X_v: np.concatenate(obs, axis=1)}
                if states is not None:
                    new_map[train_model[k].S] = states[k]
                    new_map[train_model[k].M] = masks[k]
                if num_agents > 1:
                    new_map[train_model[k].A_v] = np.concatenate([actions[j] for j in range(num_agents) if k != j], axis=1)
                td_map.update(new_map)
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def clone(lr, obs, actions):
            td_map = {}
            for k in range(num_agents):
                new_map = {train_model[k].X: obs[k], A[k]: actions[k], LR: lr}
                td_map.update(new_map)
            return sess.run(
                [neglogpac, _clone],
                td_map
            )[:-1]

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
        self.train_model = train_model
        self.act_model = act_model
        self.clone = clone

        def step(obs, acs, *_args, **_kwargs):
            a, v, neglogp = [], [], []
            for k in range(num_agents):
                a_, v_, _, neglogp_ = act_model[k].step(obs, acs)
                a.append(a_)
                v.append(v_)
                neglogp.append(neglogp_)
            return a, v, self.initial_state, neglogp

        def step_mean(obs, acs, *_args, **_kwargs):
            a, v, neglogp = [], [], []
            for k in range(num_agents):
                a_, v_, _, neglogp_ = act_model[k].step_mean(obs, acs)
                a.append(a_)
                v.append(v_)
                neglogp.append(neglogp_)
            return a, v, self.initial_state, neglogp

        def value(obs, acs, *_args, **_kwargs):
            return [act_model[k].value(obs, acs) for k in range(num_agents)]

        self.step = step
        self.value = value
        self.initial_state = [act_model[k].initial_state for k in range(num_agents)]
        self.save = save
        self.load = load
        self.step_mean = step_mean
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = env.num_envs
        try:
            ob_space = env.observation_space
            ac_space = env.action_space
            self.num_agents = num_agents = len(ob_space)
        except:
            ob_space = env.observation_space.spaces
            ac_space = env.action_space.spaces
            self.num_agents = num_agents = len(ob_space)
        self.obs = [np.zeros((nenv,) + ob_space[k].shape,
                             dtype=model.train_model[k].X.dtype.name) for k in range(num_agents)]
        self.actions = [np.zeros((nenv,) + ac_space[k].shape,
                             dtype=model.train_model[k].X.dtype.name) for k in range(num_agents)]
        obs = env.reset()
        for k in range(num_agents):
            self.obs[k][:] = obs[k]
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs = [[] for _ in range(self.num_agents)]
        mb_rewards = [[] for _ in range(self.num_agents)]
        mb_actions = [[] for _ in range(self.num_agents)]
        mb_values = [[] for _ in range(self.num_agents)]
        mb_dones = []
        mb_neglogpacs = [[] for _ in range(self.num_agents)]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.actions, self.states, self.dones)
            self.actions = actions.copy()
            for k in range(self.num_agents):
                mb_obs[k].append(self.obs[k].copy())
                mb_actions[k].append(actions[k])
                mb_values[k].append(values[k])
                mb_neglogpacs[k].append(neglogpacs[k])
            mb_dones.append(self.dones.copy())
            obs, rewards, dones, infos = self.env.step(actions)
            for k in range(self.num_agents):
                self.obs[k][:] = obs[k]
            self.dones = dones[0]
            # for i in range(self.env.num_envs):
            #     if self.dones[i]:
            #         for k in range(self.num_agents):
            #             self.actions[k][i] *= 0.0
            try:
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo: epinfos.append(maybeepinfo)
            except:
                pass
            for k in range(self.num_agents):
                mb_rewards[k].append(rewards[k])

        # batch of steps to batch of rollouts
        for k in range(self.num_agents):
            mb_obs[k] = np.asarray(mb_obs[k], dtype=self.obs[k].dtype)
            mb_rewards[k] = np.asarray(mb_rewards[k], dtype=np.float32)
            mb_actions[k] = np.asarray(mb_actions[k])
            mb_values[k] = np.asarray(mb_values[k], dtype=np.float32)
            mb_neglogpacs[k] = np.asarray(mb_neglogpacs[k], dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.actions, self.states, self.dones)

        # discount/bootstrap off value fn
        mb_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        mb_advs = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        for k in range(self.num_agents):
            lastgaelam = 0.0
            for t in reversed(range(self.nsteps)):
                if t == self.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues = last_values[k]
                else:
                    nextnonterminal = 1.0 - mb_dones[t+1]
                    nextvalues = mb_values[k][t+1]
                delta = mb_rewards[k][t] + self.gamma * nextvalues * nextnonterminal - mb_values[k][t]
                mb_advs[k][t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            mb_returns[k] = mb_advs[k] + mb_values[k]
        return (lsf01(mb_obs), lsf01(mb_returns),
                [sf01(mb_dones) for _ in range(self.num_agents)], lsf01(mb_actions), lsf01(mb_values), lsf01(mb_neglogpacs),
                mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()


def constfn(val):
    def f(_):
        return val
    return f


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=20, expert=None, clone_iters=None):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    try:
        ob_space = env.observation_space
        ac_space = env.action_space
        num_agents = len(ob_space)
    except:
        ob_space = env.observation_space.spaces
        ac_space = env.action_space.spaces
        num_agents = len(ob_space)
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_spaces=ob_space, ac_spaces=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    if expert:
        if clone_iters:
            for i in range(clone_iters):
                e_obs, e_actions, _, _ = expert.get_next_batch(nbatch // nminibatches)
                lld = model.clone(lr(1.0), e_obs, e_actions)
                if i % 100 == 0:
                    print([np.mean(l) for l in lld])

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []
        # if states is None: # nonrecurrent version
        # advs = [returns[k] - values[k] for k in range(num_agents)]
        # advs = [(advs[k] - advs[k].mean()) / (advs[k].std() + 1e-8) for k in range(num_agents)]
        inds = np.arange(nbatch)
        for _ in range(noptepochs):
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = ([a[mbinds] for a in arr] for arr in (obs, returns, masks, actions, values, neglogpacs))
                mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        # else: # recurrent version
        #     assert nenvs % nminibatches == 0
        #     envsperbatch = nenvs // nminibatches
        #     envinds = np.arange(nenvs)
        #     flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
        #     envsperbatch = nbatch_train // nsteps
        #     for _ in range(noptepochs):
        #         np.random.shuffle(envinds)
        #         for start in range(0, nenvs, envsperbatch):
        #             end = start + envsperbatch
        #             mbenvinds = envinds[start:end]
        #             mbflatinds = flatinds[mbenvinds].ravel()
        #             slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
        #             mbstates = states[mbenvinds]
        #             mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = [explained_variance(values[k], returns[k]) for k in range(num_agents)]
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            for k in range(num_agents):
                logger.logkv("explained_variance_{}".format(k), float(ev[k]))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                for k in range(num_agents):
                    logger.logkv(lossname + '{}'.format(k), lossval[k])
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
            env.save(savepath+'.ob_rms')
    env.close()


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
