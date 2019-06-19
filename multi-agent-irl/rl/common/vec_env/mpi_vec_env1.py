from rl.common.vec_env import VecEnv
import numpy as np

EXIT = 'EXIT'

class MpiVecEnv(VecEnv):
    def __init__(self, env, comm):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.comm = comm
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        if comm.Get_rank() != 0:
            self._slave_reset()
            while True:
                self._slave_step()

    def step(self, actions):
        a = self.comm.scatter(actions)
        ob,rew,done,info = self.env.step(a)
        if done: ob = self.env.reset()
        results = self.comm.gather((ob,rew,done,info))
        results = list(zip(*results))
        return (*map(np.array, results[:3]), results[3])

    def close(self):
        self.comm.scatter([EXIT for _ in range(self.num_envs)])

    def _slave_step(self):
        a = self.comm.scatter(None)
        if a == EXIT:
            return
        ob,rew,done,info = self.env.step(a)
        if done: ob = self.env.reset()
        self.comm.gather((ob,rew,done,info))

    def reset(self):
        return np.array(self.comm.gather(self.env.reset()))

    def _slave_reset(self):
        self.comm.gather(self.env.reset())

    @property
    def num_envs(self):
        return self.comm.Get_size()
