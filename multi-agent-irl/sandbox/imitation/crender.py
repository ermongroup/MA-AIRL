import gym
import click
import rl.common.tf_util as U
from sandbox.ppo_sgd import cmlp_policy
import multiagent
import make_env
import gym.spaces
import numpy as np
import time


@click.command()
@click.option('--path', type=click.STRING,
              # default='/Users/jiaming/atlas/exps/cmappo-sgd/simple_push/l-0.0001-b-10000/seed-1'
              default='/tmp/exps/cmappo-sgd/simple_speaker_listener/l-0.0001-b-10000/seed-1'
              )
def render(path):
    def policy_fn(name, ob_space, ac_space, index, all_ob_space):
        pi = cmlp_policy.MlpPolicy(
            name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, index=index, all_ob_space=all_ob_space
        )
        return pi

    env = make_env.make_env('simple_speaker_listener')
    ob_space = env.observation_space
    ac_space = env.action_space
    all_ob_shape = 0
    for k in range(env.n):
        all_ob_shape += ob_space[k].shape[0]
    all_ob_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(all_ob_shape,))
    pi = []
    for k in range(env.n):
        pi.append(policy_fn("pi_%d"%k, ob_space[k], ac_space[k], # gym.spaces.Box(low=-1., high=1., shape=(5,)),
                            k, all_ob_space))
    sess = U.single_threaded_session()
    sess.__enter__()

    for k in range(env.n):
        pi[k].restore(path + '/model_%d.ckpt'%k)
        # print(U.get_session().run(pi[k].ob_rms.mean))

    obs = env.reset()
    done = False

    #print(obs.shape)

    for i in range(100):
        obs = env.reset()
        step = 0
        done = False
        while not done:
            action = []
            flattened_obs = []
            for j in obs:
                flattened_obs.extend(j)
            for k in range(env.n):
                act, _ = pi[k].act(False, obs[k], np.array(flattened_obs))
                action.append(act)
            # print(action)
            obs, rew, done, _ = env.step(action)
            print(rew)
            step += 1
            env.render()
            time.sleep(0.1)
            if step == 50 or True in done:
                done = True
                step = 0
            else:
                done = False

if __name__ == '__main__':
    render()