import tensorflow as tf
import numpy as np
import joblib
from rl.acktr.utils import Scheduler, find_trainable_variables
from rl.acktr.utils import fc, mse
from rl.acktr import kfac

disc_types = ['decentralized', 'centralized', 'single']


class Discriminator(object):
    def __init__(self, sess, ob_spaces, ac_spaces,
                 nstack, index, disc_type='decentralized', hidden_size=128,
                 lr_rate=0.01, total_steps=50000, scope="discriminator", kfac_clip=0.001, max_grad_norm=0.5):
        self.lr = Scheduler(v=lr_rate, nvalues=total_steps, schedule='linear')
        self.disc_type = disc_type
        if disc_type not in disc_types:
            assert False
        self.scope = scope
        self.index = index
        self.sess = sess
        ob_space = ob_spaces[index]
        ac_space = ac_spaces[index]
        self.ob_shape = ob_space.shape[0] * nstack
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
            input_shape = self.ob_shape + self.ac_shape
        elif disc_type == 'centralized':
            input_shape = self.all_ob_shape + self.all_ac_shape
        elif disc_type == 'single':
            input_shape = self.all_ob_shape + self.all_ac_shape
        else:
            assert False

        self.g = tf.placeholder(tf.float32, (None, input_shape))
        self.e = tf.placeholder(tf.float32, (None, input_shape))
        self.lr_rate = tf.placeholder(tf.float32, ())
        self.adv = tf.placeholder(tf.float32, ())

        num_outputs = len(ob_spaces) if disc_type == 'centralized' else 1

        logits = self.build_graph(tf.concat([self.g, self.e], axis=0), num_outputs, reuse=False)
        labels = tf.concat([tf.zeros([tf.shape(self.g)[0], 1]), tf.ones([tf.shape(self.e)[0], 1])], axis=0)

        g_logits = self.build_graph(self.g, num_outputs, reuse=True)
        e_logits = self.build_graph(self.e, num_outputs, reuse=True)

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=g_logits, labels=tf.zeros_like(g_logits)))
        self.e_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=e_logits, labels=tf.ones_like(e_logits)))

        self.total_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        fisher_loss = -self.total_loss

        # self.reward_op = tf.sigmoid(g_logits) * 2.0 - 1
        self.reward_op = tf.log(tf.sigmoid(g_logits) + 1e-10)

        # self.reward_op = tf.nn.sigmoid_cross_entropy_with_logits(logits=g_logits, labels=tf.zeros_like(g_logits))

        self.var_list = self.get_trainable_variables()
        params = find_trainable_variables(self.scope)
        grads = tf.gradients(self.total_loss, params)

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

    def build_graph(self, x, num_outputs=1, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            p_h1 = fc(x, 'fc1', nh=self.hidden_size)
            p_h2 = fc(p_h1, 'fc2', nh=self.hidden_size)
            logits = fc(p_h2, 'out', nh=num_outputs, act=lambda x: x)
        return logits

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
        feed_dict = {self.g: np.concatenate([obs, acs], axis=1)}
        return self.sess.run(self.reward_op, feed_dict)

    def train(self, g_obs, g_acs, e_obs, e_acs):
        feed_dict = {self.g: np.concatenate([g_obs, g_acs], axis=1),
                     self.e: np.concatenate([e_obs, e_acs], axis=1), self.lr_rate: self.lr.value()}
        loss, _ = self.sess.run([self.total_loss, self.d_optim], feed_dict)
        g_loss, e_loss = self.sess.run([self.g_loss, self.e_loss], feed_dict)
        return g_loss, e_loss, None, None

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

