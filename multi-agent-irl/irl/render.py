import gym
import click
import multiagent
import time
import tensorflow as tf
import make_env
import numpy as np
from rl.common.misc_util import set_global_seeds
from sandbox.mack.acktr_disc import Model, onehot
from sandbox.mack.policies import CategoricalPolicy
from rl import bench
import imageio
import pickle as pkl


@click.command()
@click.option('--env', type=click.STRING)
@click.option('--image', is_flag=True, flag_value=True)
def render(env, image):
    tf.reset_default_graph()

    env_id = env

    def create_env():
        env = make_env.make_env(env_id)
        env.seed(10)
        # env = bench.Monitor(env, '/tmp/',  allow_early_resets=True)
        set_global_seeds(10)
        return env

    env = create_env()
    path = '/atlas/u/lantaoyu/exps/airl/simple_spread/decentralized/s-200/l-0.1-b-1000-d-0.1-c-500-l2-0.1-iter-1-r-0.0/seed-2/m_24000'

    print(path)
    n_agents = len(env.action_space)

    ob_space = env.observation_space
    ac_space = env.action_space

    print('observation space')
    print(ob_space)
    print('action space')
    print(ac_space)

    n_actions = [action.n for action in ac_space]

    make_model = lambda: Model(
        CategoricalPolicy, ob_space, ac_space, 1, total_timesteps=1e7, nprocs=2, nsteps=500,
        nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.01, max_grad_norm=0.5, kfac_clip=0.001,
        lrschedule='linear', identical=make_env.get_identical(env_id), use_kfac=False)
    model = make_model()
    print("load model from", path)
    model.load(path)

    images = []
    sample_trajs = []
    num_trajs = 100
    max_steps = 50
    avg_ret = [[] for _ in range(n_agents)]

    for i in range(num_trajs):
        all_ob, all_agent_ob, all_ac, all_rew, ep_ret = [], [], [], [], [0 for k in range(n_agents)]
        for k in range(n_agents):
            all_ob.append([])
            all_ac.append([])
            all_rew.append([])
        obs = env.reset()
        obs = [ob[None, :] for ob in obs]
        action = [np.zeros([1]) for _ in range(n_agents)]
        step = 0
        done = False
        while not done:
            action, _, _ = model.step(obs, action)
            actions_list = [onehot(action[k][0], n_actions[k]) for k in range(n_agents)]
            # actions_list = [onehot(np.random.randint(n_actions[k]), n_actions[k]) for k in range(n_agents)]

            for k in range(n_agents):
                all_ob[k].append(obs[k])
                all_ac[k].append(actions_list[k])
            all_agent_ob.append(np.concatenate(obs, axis=1))
            obs, rew, done, _ = env.step(actions_list)
            for k in range(n_agents):
                all_rew[k].append(rew[k])
                ep_ret[k] += rew[k]
            obs = [ob[None, :] for ob in obs]
            step += 1

            if image:
                img = env.render(mode='rgb_array')
                images.append(img[0])
                time.sleep(0.02)
            if step == max_steps or True in done:
                done = True
                step = 0
            else:
                done = False

        for k in range(n_agents):
            all_ob[k] = np.squeeze(all_ob[k])

        all_agent_ob = np.squeeze(all_agent_ob)
        traj_data = {
            "ob": all_ob, "ac": all_ac, "rew": all_rew,
            "ep_ret": ep_ret, "all_ob": all_agent_ob
        }

        sample_trajs.append(traj_data)
        # print('traj_num', i, 'expected_return', ep_ret)

        for k in range(n_agents):
            avg_ret[k].append(ep_ret[k])

    print(path)
    for k in range(n_agents):
        print('agent', k, np.mean(avg_ret[k]), np.std(avg_ret[k]))

    images = np.array(images)
    # pkl.dump(sample_trajs, open(path + '-%dtra.pkl' % num_trajs, 'wb'))
    if image:
        print(images.shape)
        imageio.mimsave(path + '.mp4', images, fps=25)


if __name__ == '__main__':
    render()
