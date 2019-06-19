from baselines.common.vec_env import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd
import numpy as np
import joblib


class VecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)


class MAVecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        try:
            self.num_agents = num_agents = len(self.observation_space)
            self.ob_rms = [RunningMeanStd(shape=self.observation_space[k].shape) for k in range(num_agents)] if ob else None
        except:
            self.num_agents = num_agents = len(self.observation_space.spaces)
            self.ob_rms = [RunningMeanStd(shape=self.observation_space.spaces[k].shape) for k in range(num_agents)] if ob else None

        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        #[RunningMeanStd(shape=()) for k in range(num_agents)] if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        # self.ret = [np.zeros(self.num_envs) for _ in range(num_agents)]
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        # print(rews)
        self.ret = self.ret * self.gamma + rews[0]
        # self.ret = [self.ret[k] * self.gamma + rews[k] for k in range(self.num_agents)]
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            # for k in range(self.num_agents):
            # print(self.ret_rms.mean, self.ret_rms.var)
            rews = [np.clip(rews[k] / np.sqrt(self.ret_rms.var + self.epsilon),
                            -self.cliprew, self.cliprew) for k in range(self.num_agents)]
            # print('---')
            # print(rews)
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            for k in range(self.num_agents):
                self.ob_rms[k].update(obs[k])
            obs = [np.clip((obs[k] - self.ob_rms[k].mean) / np.sqrt(self.ob_rms[k].var + self.epsilon), -self.clipob, self.clipob) for k in range(self.num_agents)]
            return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)

    def save(self, path):
        joblib.dump(self.ob_rms, path)

    def load(self, path):
        self.ob_rms = joblib.load(path)