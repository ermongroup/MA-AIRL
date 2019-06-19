#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from rl import make_env
from rl import bench, logger
import os
import robosumo.envs
from irl.dataset import MADataSet


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def train(env_id, num_timesteps, seed, num_cpu, batch, lr):
    from rl.common import set_global_seeds
    from rl.common.vec_env.vec_normalize import MAVecNormalize
    from rl.common.ma_wrappers import MAWrapper
    from sandbox.mppo import ppo2
    from sandbox.mppo.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def _make_env(rank):
        env = gym.make('RoboSumo-Ant-vs-Ant-v0')
        env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
        return env

    env = SubprocVecEnv([lambda: _make_env(i) for i in range(num_cpu)], is_multi_agent=True)
    env = MAVecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    expert = MADataSet('/atlas/u/tsong/Projects/imitation/ant-vs-ant.pkl')
    ppo2.learn(policy=policy, env=env, nsteps=batch // num_cpu, nminibatches=160,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=lr,
        cliprange=0.2,
        total_timesteps=num_timesteps, expert=expert, clone_iters=1000)


def main():
    parser = mujoco_arg_parser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch', type=int, default=2048)
    args = parser.parse_args()
    logdir = '/atlas/u/tsong/exps/mppo/sumo' + '/l-{}-b-{}/seed-{}'.format(args.lr, args.batch, args.seed)
    try:
        logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])
    except:
        logger.configure()
    train(args.env, num_timesteps=1e7, seed=args.seed, num_cpu=args.cpu, batch=args.batch, lr=args.lr)


if __name__ == '__main__':
    main()
