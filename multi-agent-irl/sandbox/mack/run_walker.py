#!/usr/bin/env python
import logging
import os
import itertools
import click
import gym
from rl import bench
from rl import logger
from rl.common import set_global_seeds
from rl import make_env
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
from sandbox.mack.acktr_cont import learn
from sandbox.mack.policies import GaussianPolicy

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu, gae):
    def create_env(rank):
        def _thunk():
            env = gym.make('BipedalWalker-v2') # make_env(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])

    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=False)
    policy_fn = GaussianPolicy
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.00, gae=gae)
    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='/tmp')
@click.option('--env', type=click.STRING, default='biwalker1')
@click.option('--lr', type=click.FLOAT, default=0.03)
@click.option('--seed', type=click.INT, default=1)
@click.option('--batch_size', type=click.INT, default=2400)
@click.option('--atlas', is_flag=True, flag_value=True)
@click.option('--gae', is_flag=True, flag_value=True)
def main(logdir, env, lr, seed, batch_size, atlas, gae):
    env_ids = [env]
    lrs = [lr]
    seeds = [seed]
    batch_sizes = [batch_size]

    if atlas:
        logdir = '/atlas/u/tsong'
    print('logging to: ' + logdir)

    gae_str = '/gae' if gae else '/a3c'

    for env_id, seed, lr, batch_size in itertools.product(env_ids, seeds, lrs, batch_sizes):
        train(logdir + '/exps/mack/' + env_id + '{}/l-{}-b-{}/seed-{}'.format(gae_str, lr, batch_size, seed),
              env_id, 5e7, lr, batch_size, seed, 1, True)


if __name__ == "__main__":
    main()
