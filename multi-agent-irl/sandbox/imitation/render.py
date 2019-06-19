import gym
import click
import rl.common.tf_util as U
from sandbox.ppo_sgd import mlp_policy
#from gym.envs.registration import register

'''
register(
    id='Walker2d-v5',
    entry_point='rl.envs.mujoco_env.walker2d:Walker2dEnv',
    kwargs={'target_vel': 5.}
)
'''

import multiagent
import make_env
import gym.spaces
import numpy as np

@click.command()
@click.option('--path', type=click.STRING,
              default='./log/exps/mappo-sgd/simple_tag/l-0.0003-b-2048/seed-1')
def render(path):
    def policy_fn(name, ob_space, ac_space, id):
        pi = mlp_policy.MlpPolicy(
            name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, id = id
        )
        return pi

    env = make_env.make_env('Walker2d-v5')
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = []
    for k in range(env.n):
        pi.append(policy_fn("pi_%d"%k, ob_space[k], gym.spaces.Box(low=-1., high=1., shape=(5,)), k))
    sess = U.single_threaded_session()
    sess.__enter__()

    for k in range(env.n):
        pi[k].restore(path + '/model_%d.ckpt'%k)
        print(U.get_session().run(pi[k].ob_rms.mean))

    obs = env.reset()
    done = False

    #print(obs.shape)

    for i in range(100):
        obs = env.reset()
        step = 0
        done = False
        while not done:
            action = []
            for k in range(env.n):
                act, _ = pi[k].act(False, obs[k])
                action.append(act)
            # print(action)
            obs, _, done, _ = env.step(action)
            step += 1
            env.render()
            if step == 100:
                done = True
            else:
                done = False

if __name__ == '__main__':
    render()