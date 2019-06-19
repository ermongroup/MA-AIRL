import numpy as np
from . import VecEnv


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns, swi=False, is_multi_agent=False):
        envswi = [fn() for fn in env_fns]
        if swi:
            self.envs, self.switchers = zip(*envswi)
        else:
            self.envs = envswi
        env = self.envs[0]
        self.action_space = env.action_space
        self.observation_space = env.observation_space        
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.is_multi_agent = is_multi_agent
        self.num_agents = len(env.observation_space) if is_multi_agent else 1

    def switch_to(self,i=None,option=None):
        assert self.switchers
        if i:
            for switcher in self.switchers:
                if option:
                    switcher.switch_to(i,option=option)
                else:
                    switcher.switch_to(i)
            self.i = i
        else:
            assert self.i
            self.i +=1
            for switcher in self.switchers:
                if option:
                    switcher.switch_to(self.i, option=option)
                else:
                    switcher.switch_to(self.i)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        action_n = self.actions
        if self.is_multi_agent:
            action_n = [[ac[i] for ac in action_n] for i in range(len(self.envs))]
            results = [env.step(a) for (a,env) in zip(action_n, self.envs)]
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
            results = [env.step(a) for (a,env) in zip(action_n, self.envs)]
            obs, rews, dones, infos = map(np.array, zip(*results))
            self.ts += 1
            for (i, done) in enumerate(dones):
                if done:
                    obs[i] = self.envs[i].reset()
                    self.ts[i] = 0
            return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        if self.is_multi_agent:
            obs = [[result[k] for result in results] for k in range(self.num_agents)]
            obs = [np.stack(ob) for ob in obs]
            return obs
        else:
            return np.array(results)

    @property
    def num_envs(self):
        return len(self.envs)

    def render(self, mode):
        return self.envs[0].render()
