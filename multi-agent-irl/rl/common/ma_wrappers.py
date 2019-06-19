import numpy as np
from collections import deque
from PIL import Image
import gym
from gym import spaces


class MAWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = [self.env.observation_space]
        self.action_space = [self.env.action_space]
        self.n = 1

    def step(self, action):
        obs, reward, done, info = self.env.step(action[0])
        return [obs], [reward], [done], info

    def reset(self):
        return [self.env.reset()]
