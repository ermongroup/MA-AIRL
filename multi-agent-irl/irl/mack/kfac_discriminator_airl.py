import tensorflow as tf
import numpy as np
import joblib
from rl.acktr.utils import Scheduler, find_trainable_variables
from rl.acktr.utils import fc, mse
from rl.acktr import kfac
from irl.mack.tf_util import relu_layer, linear, tanh_layer

disc_types = ['decentralized', 'centralized', 'single', 'decentralized-all']


class Discriminator(object):
    def __init__(self, sess, ob_spaces, ac_spaces, state_only, discount,
                 nstack, index, disc_type='decentralized', hidden_size=128,
                 lr_rate=0.01, total_steps=50000, scope="discriminator", kfac_clip=0.001, max_grad_norm=0.5,
                 l2_loss_ratio=0.01):
        self.lr = Scheduler(v=lr_rate, nvalues=total_steps, schedule='linear')
        self.disc_type = disc_type
        self.l2_loss_ratio = l2_loss_ratio
        if disc_type not in disc_types:
            assert False
        self.state_only = state_only
        self.gamma = discount
        self.scope = scope
        self.index = index
        self.sess = sess
        ob_space = ob_spaces[index]
        ac_space = ac_spaces[index]
        self.ob_shape = ob_space.shape[0] * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        try:
            nact = ac_space.n
        except:
            nact = ac_space.shape[0]
        self.ac_shape = nact * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        try:
            self.all_ac_shape = sum([ac.n for ac in ac_spaces]) * nstack
        except:
            self.all_ac_shape = sum([ac.shape[0] for ac in ac_spaces]) * nstack
        self.hidden_size = hidden_size

        if disc_type == 'decentralized':
            self.obs = tf.placeholder(tf.float32, (None, self.ob_shape))
            self.nobs = tf.placeholder(tf.float32, (None, self.ob_shape))
            self.act = tf.placeholder(tf.float32, (None, self.ac_shape))
            self.labels = tf.placeholder(tf.float32, (None, 1))
            self.lprobs = tf.placeholder(tf.float32, (None, 1))
        elif disc_type == 'decentralized-all':
            self.obs = tf.placeholder(tf.float32, (None, self.all_ob_shape))
            self.nobs = tf.placeholder(tf.float32, (None, self.all_ob_shape))
            self.act = tf.placeholder(tf.float32, (None, self.all_ac_shape))
            self.labels = tf.placeholder(tf.float32, (None, 1))
            self.lprobs = tf.placeholder(tf.float32, (None, 1))
        else:
            assert False

        self.lr_rate = tf.placeholder(tf.float32, ())

        with tf.variable_scope(self.scope):
            rew_input = self.obs
            if not self.state_only:
                rew_input = tf.concat([self.obs, self.act], axis=1)

            with tf.variable_scope('reward'):
                self.reward = self.relu_net(rew_input, dout=1)
                # self.reward = self.tanh_net(rew_input, dout=1)

            with tf.variable_scope('vfn'):
                self.value_fn_n = self.relu_net(self.nobs, dout=1)
                # self.value_fn_n = self.tanh_net(self.nobs, dout=1)
            with tf.variable_scope('vfn', reuse=True):
                self.value_fn = self.relu_net(self.obs, dout=1)
                # self.value_fn = self.tanh_net(self.obs, dout=1)

            log_q_tau = self.lprobs
            log_p_tau = self.reward + self.gamma * self.value_fn_n - self.value_fn
            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.discrim_output = tf.exp(log_p_tau - log_pq)

        self.total_loss = -tf.reduce_mean(self.labels * (log_p_tau - log_pq) + (1 - self.labels) * (log_q_tau - log_pq))
        self.var_list = self.get_trainable_variables()
        params = find_trainable_variables(self.scope)
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params]) * self.l2_loss_ratio
        self.total_loss += self.l2_loss

        grads = tf.gradients(self.total_loss, params)
        # fisher_loss = -self.total_loss
        # self.d_optim = tf.train.AdamOptimizer(self.lr_rate, beta1=0.5, beta2=0.9).minimize(self.total_loss, var_list=self.var_list)
        with tf.variable_scope(self.scope + '/d_optim'):
            # d_optim = kfac.KfacOptimizer(
            #     learning_rate=self.lr_rate, clip_kl=kfac_clip,
            #     momentum=0.9, kfac_update=1, epsilon=0.01,
            #     stats_decay=0.99, async=0, cold_iter=10,
            #     max_grad_norm=max_grad_norm)
            # update_stats_op = d_optim.compute_and_apply_stats(fisher_loss, var_list=params)
            # train_op, q_runner = d_optim.apply_gradients(list(zip(grads, params)))
            # self.q_runner = q_runner
            d_optim = tf.train.AdamOptimizer(learning_rate=self.lr_rate)
            train_op = d_optim.apply_gradients(list(zip(grads, params)))
        self.d_optim = train_op
        self.saver = tf.train.Saver(self.get_variables())

        self.params_flat = self.get_trainable_variables()

    def relu_net(self, x, layers=2, dout=1, hidden_size=128):
        out = x
        for i in range(layers):
            out = relu_layer(out, dout=hidden_size, name='l%d' % i)
        out = linear(out, dout=dout, name='lfinal')
        return out

    def tanh_net(self, x, layers=2, dout=1, hidden_size=128):
        out = x
        for i in range(layers):
            out = tanh_layer(out, dout=hidden_size, name='l%d' % i)
        out = linear(out, dout=dout, name='lfinal')
        return out

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs, obs_next, path_probs, discrim_score=False):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
        if discrim_score:
            feed_dict = {self.obs: obs,
                         self.act: acs,
                         self.nobs: obs_next,
                         self.lprobs: path_probs}
            scores = self.sess.run(self.discrim_output, feed_dict)
            score = np.log(scores + 1e-20) - np.log(1 - scores + 1e-20)
        else:
            feed_dict = {self.obs: obs,
                         self.act: acs}
            score = self.sess.run(self.reward, feed_dict)
        return score

    def train(self, g_obs, g_acs, g_nobs, g_probs, e_obs, e_acs, e_nobs, e_probs):
        labels = np.concatenate((np.zeros([g_obs.shape[0], 1]), np.ones([e_obs.shape[0], 1])), axis=0)
        feed_dict = {self.obs: np.concatenate([g_obs, e_obs], axis=0),
                     self.act: np.concatenate([g_acs, e_acs], axis=0),
                     self.nobs: np.concatenate([g_nobs, e_nobs], axis=0),
                     self.lprobs: np.concatenate([g_probs, e_probs], axis=0),
                     self.labels: labels,
                     self.lr_rate: self.lr.value()}
        loss, _ = self.sess.run([self.total_loss, self.d_optim], feed_dict)
        return loss

    def restore(self, path):
        print('restoring from:' + path)
        self.saver.restore(self.sess, path)

    def save(self, save_path):
        ps = self.sess.run(self.params_flat)
        joblib.dump(ps, save_path)

    def load(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        for p, loaded_p in zip(self.params_flat, loaded_params):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)
