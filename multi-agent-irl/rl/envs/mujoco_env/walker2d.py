import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, target_vel):
        self.target_vel = target_vel
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        print(reward)
        # reward = -(reward - self.target_vel) ** 2

        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (2.0 > height > 0.8 and 1.0 > ang > -1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20


if __name__ == '__main__':
    try:
        register(
            id='NewWalker2d_5-v1',
            entry_point='new_walker2d_5:NewWalker2dEnv',
            kwargs={'target_vel': 5.}
        )
    except gym.error.Error:
        pass
    env = gym.make('NewWalker2d_5-v1')
