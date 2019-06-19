#!/usr/bin/env python
import itertools
import logging
import os.path as osp
import os

import click
import gym
import ray

os.environ['CUDA_VISIBLE_DEVICES'] = ''


@ray.remote(num_cpus=1)
def train(logdir, env_id, lr, num_timesteps, seed, timesteps_per_batch, cont=False):
    from sandbox.ppo_sgd import mlp_policy
    from sandbox.ppo_sgd import pposgd_simple
    from rl import logger
    from rl.common import set_global_seeds, tf_util as U
    from rl import bench

    from gym.envs.registration import register
    import multiagent
    import make_env

    logger.configure(logdir, format_strs=['log', 'json', 'tensorboard'])
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = make_env.make_env(env_id)

    def policy_fn(name, ob_space, ac_space, id):
        pi = mlp_policy.MlpPolicy(
            name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, id=id
        )
        return pi

    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(
        env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_batch=timesteps_per_batch,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=lr, optim_batchsize=64,
        gamma=0.99, lam=0.95, schedule='linear', cont=cont
    )
    env.close()
    return None


@click.command()
@click.option('--logdir', default='/tmp', type=click.STRING)
@click.option('--cont', is_flag=True, flag_value=True)
def main(logdir, cont):
    env_ids = [
        'simple_tag'
    ]
    lrs = [
        0.0003
    ]
    seeds = [1,2,3,4]
    batch_sizes = [2048]

    num_cpus = len(env_ids) * len(lrs) * len(seeds) * len(batch_sizes)
    print(len(env_ids), len(lrs) , len(seeds) , len(batch_sizes))
    ray.init(num_cpus=num_cpus, num_gpus=0)
    print('Requesting {} cpus.'.format(num_cpus))

    jobs = [
        train.remote(
            logdir + '/exps/mappo-sgd/' + env_id + '/l-{}-b-{}/seed-{}'.format(lr, batch_size, seed),
            env_id, lr, 1e7, seed, batch_size, cont)
        for env_id, lr, batch_size, seed in itertools.product(env_ids, lrs, batch_sizes, seeds)
    ]

    print(jobs)
    ray.get(jobs)


if __name__ == '__main__':
    main()
