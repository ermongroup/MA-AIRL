import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, env_fn_wrapper, is_multi_agent):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if is_multi_agent:
                if done[0]:
                    ob = env.reset()
            else:
                if done:
                    ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        elif cmd == 'render':
            env.render()
            remote.send(0)
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, is_multi_agent=False):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])        
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn), is_multi_agent))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()
        self.is_multi_agent = is_multi_agent
        self.num_agents = None
        if is_multi_agent:
            try:
                n = len(self.action_space)
            except:
                n = len(self.action_space.spaces)
            self.num_agents = n

    def step_async(self, actions):
        # if self.is_multi_agent:
        #     remote_action = []
        #     for i in range(len(self.remotes)):
        #         remote_action.append([action[i] for action in actions])
        #     actions = remote_action

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        if self.is_multi_agent:
            obs, rews, dones, infos = [], [], [], []
            for k in range(self.num_agents):
                obs.append([result[0][k] for result in results])
                rews.append([result[1][k] for result in results])
                dones.append([result[2][k] for result in results])
            try:
                infos = [result[3] for result in results]
            except:
                infos = None

            obs = [np.stack(ob) for ob in obs]
            rews = [np.stack(rew) for rew in rews]
            dones = [np.stack(done) for done in dones]
            return obs, rews, dones, infos
        else:
            obs, rews, dones, infos = zip(*results)
            return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        if self.is_multi_agent:
            results = [remote.recv() for remote in self.remotes]
            obs = [[result[k] for result in results] for k in range(self.num_agents)]
            obs = [np.stack(ob) for ob in obs]
            return obs
        else:
            return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)



if __name__ == '__main__':
    from make_env import make_env

    def create_env(rank):
        def _thunk():
            env = make_env('simple_push')
            env.seed(rank)
            return env
        return _thunk

    env = SubprocVecEnv([create_env(i) for i in range(0, 4)], is_multi_agent=True)
    env.reset()
    obs, rews, dones, _ = env.step(
        [[np.array([0, 1, 0, 0, 0]), np.array([2, 0, 0, 0, 0])] for _ in range(4)]
    )
    print(env.observation_space)
    print(obs)
    print(rews[0].shape)
    print(dones[1].shape)
    env.close()
