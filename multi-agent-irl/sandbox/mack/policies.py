import numpy as np
import tensorflow as tf

import rl.common.tf_util as U
from rl.acktr.utils import conv, fc, dense, conv_to_fc, sample, kl_div


class CategoricalPolicy(object):
    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.n
        actions = tf.placeholder(tf.int32, (nbatch))
        all_ac_shape = (nbatch, (sum([ac.n for ac in ac_spaces]) - nact) * nstack)
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)
        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=128, init_scale=np.sqrt(2))
            h2 = fc(h1, 'fc2', nh=128, init_scale=np.sqrt(2))
            pi = fc(h2, 'pi', nact, act=lambda x: x)

        with tf.variable_scope('value_{}'.format(name), reuse=reuse):
            if len(ob_spaces) > 1:
                Y = tf.concat([X_v, A_v], axis=1)
            else:
                Y = X_v
            h3 = fc(Y, 'fc3', nh=256, init_scale=np.sqrt(2))
            h4 = fc(h3, 'fc4', nh=256, init_scale=np.sqrt(2))
            vf = fc(h4, 'v', 1, act=lambda x: x)

        self.log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi, labels=actions)
        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = []  # not stateful

        def step_log_prob(ob, acts):
            log_prob = sess.run(self.log_prob, {X: ob, actions: acts})
            return log_prob.reshape([-1, 1])

        def step(ob, obs, a_v, *_args, **_kwargs):
            if a_v is not None:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs, A_v: a_v})
            else:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs})
            return a, v, []  # dummy state

        def value(ob, a_v, *_args, **_kwargs):
            if a_v is not None:
                return sess.run(v0, {X_v: ob, A_v: a_v})
            else:
                return sess.run(v0, {X_v: ob})

        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.vf = vf
        self.step_log_prob = step_log_prob
        self.step = step
        self.value = value


class GaussianPolicy(object):
    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.shape[0]
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack)
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)
        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            h2 = fc(h1, 'fc2', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            pi = fc(h2, 'pi', nact, act=lambda x: x, init_scale=0.01)

        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            logstd = tf.get_variable('sigma', shape=[nact], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
            logstd = tf.expand_dims(logstd, 0)
            std = tf.exp(logstd)
            std = tf.tile(std, [nbatch, 1])

        with tf.variable_scope('value_{}'.format(name), reuse=reuse):
            if len(ob_spaces) > 1:
                Y = tf.concat([X_v, A_v], axis=1)
            else:
                Y = X_v
            h3 = fc(Y, 'fc3', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            h4 = fc(h3, 'fc4', nh=64, init_scale=np.sqrt(2), act=tf.nn.tanh)
            vf = fc(h4, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = pi + tf.random_normal(tf.shape(std), 0.0, 1.0) * std

        self.initial_state = []  # not stateful

        def step(ob, obs, a_v, *_args, **_kwargs):
            if a_v is not None:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs, A_v: a_v})
            else:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs})
            return a, v, []  # dummy state

        def value(ob, a_v, *_args, **_kwargs):
            if a_v is not None:
                return sess.run(v0, {X_v: ob, A_v: a_v})
            else:
                return sess.run(v0, {X_v: ob})

        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.vf = vf
        self.std = std
        self.logstd = logstd
        self.step = step
        self.value = value
        self.mean_std = tf.concat([pi, std], axis=1)


class MultiCategoricalPolicy(object):
    def __init__(self, sess, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, reuse=False, name='model'):
        nbins = 11
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.shape[0]
        all_ac_shape = (nbatch, (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack)
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        X_v = tf.placeholder(tf.float32, all_ob_shape)
        A_v = tf.placeholder(tf.float32, all_ac_shape)
        with tf.variable_scope('policy_{}'.format(name), reuse=reuse):
            h1 = fc(X, 'fc1', nh=128, init_scale=np.sqrt(2))
            h2 = fc(h1, 'fc2', nh=128, init_scale=np.sqrt(2))
            pi = fc(h2, 'pi', nact * nbins, act=lambda x: x)

        with tf.variable_scope('value_{}'.format(name), reuse=reuse):
            if len(ob_spaces) > 1:
                Y = tf.concat([X_v, A_v], axis=1)
            else:
                Y = X_v
            h3 = fc(Y, 'fc3', nh=256, init_scale=np.sqrt(2))
            h4 = fc(h3, 'fc4', nh=256, init_scale=np.sqrt(2))
            vf = fc(h4, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        pi = tf.reshape(pi, [nbatch, nact, nbins])
        a0 = sample(pi, axis=2)
        self.initial_state = []  # not stateful

        def step(ob, obs, a_v, *_args, **_kwargs):
            # output continuous actions within [-1, 1]
            if a_v is not None:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs, A_v: a_v})
            else:
                a, v = sess.run([a0, v0], {X: ob, X_v: obs})
            a = transform(a)
            return a, v, []  # dummy state

        def value(ob, a_v, *_args, **_kwargs):
            if a_v is not None:
                return sess.run(v0, {X_v: ob, A_v: a_v})
            else:
                return sess.run(v0, {X_v: ob})

        def transform(a):
            # transform from [0, 9] to [-0.8, 0.8]
            a = np.array(a, dtype=np.float32)
            a = (a - (nbins - 1) / 2) / (nbins - 1) * 2.0
            return a

        self.X = X
        self.X_v = X_v
        self.A_v = A_v
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value