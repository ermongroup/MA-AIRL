#!/usr/bin/env python
import itertools
import logging
import os.path as osp
import os

import click
import gym
import ray

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@ray.remote(num_cpus=1)
def train(logdir, env_id, lr, num_timesteps, seed, timesteps_per_batch, cont=False):
    from sandbox.ppo_sgd import cmlp_policy
    from sandbox.ppo_sgd import cmappo_simple
    from rl import logger
    from rl.common import set_global_seeds, tf_util as U
    from rl import bench

    from gym.envs.registration import register
    import multiagent
    import make_env

    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])

    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = make_env.make_env(env_id)

    def policy_fn(name, ob_space, ac_space, index, all_ob_space):
        pi = cmlp_policy.MlpPolicy(
            name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, index=index, all_ob_space=all_ob_space
        )
        return pi

    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    cmappo_simple.learn(
        env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_batch=timesteps_per_batch,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=4, optim_stepsize=lr, optim_batchsize=64,
        gamma=0.95, lam=0.95, schedule='linear', cont=cont
    )
    env.close()
    return None


@click.command()
@click.option('--logdir', default='/tmp', type=click.STRING)
@click.option('--cont', is_flag=True, flag_value=True)
def main(logdir, cont):
    env_ids = [
        'simple_speaker_listener'
    ]
    lrs = [
        0.0001 # 0.0001, 0.003, 0.0005, 0.0001
    ]
    seeds = [1]
    batch_sizes = [50000]

    num_cpus = len(env_ids) * len(lrs) * len(seeds) * len(batch_sizes)
    # print(len(env_ids), len(lrs) , len(seeds) , len(batch_sizes))
    ray.init(num_cpus=num_cpus, num_gpus=0)
    print('Requesting {} cpus.'.format(num_cpus))

    jobs = [
        train.remote(
            logdir + '/exps/cmappo-sgd/' + env_id + '/l-{}-b-{}/seed-{}'.format(lr, batch_size, seed),
            env_id, lr, 1e7, seed, batch_size, cont)
        for env_id, lr, batch_size, seed in itertools.product(env_ids, lrs, batch_sizes, seeds)
    ]

    print(jobs)
    ray.get(jobs)


if __name__ == '__main__':
    main()
