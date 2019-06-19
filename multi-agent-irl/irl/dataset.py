import pickle as pkl
import numpy as np
from rl import logger
from tqdm import tqdm


class Dset(object):
    def __init__(self, inputs, labels, nobs, all_obs, rews, randomize, num_agents, nobs_flag=False):
        self.inputs = inputs.copy()
        self.labels = labels.copy()
        self.nobs_flag = nobs_flag
        if nobs_flag:
            self.nobs = nobs.copy()
        self.all_obs = all_obs.copy()
        self.rews = rews.copy()
        self.num_agents = num_agents
        assert len(self.inputs[0]) == len(self.labels[0])
        self.randomize = randomize
        self.num_pairs = len(inputs[0])
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            for k in range(self.num_agents):
                self.inputs[k] = self.inputs[k][idx, :]
                self.labels[k] = self.labels[k][idx, :]
                if self.nobs_flag:
                    self.nobs[k] = self.nobs[k][idx, :]
                self.rews[k] = self.rews[k][idx]
            self.all_obs = self.all_obs[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels, self.all_obs, self.rews
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs, labels, rews, nobs = [], [], [], []
        for k in range(self.num_agents):
            inputs.append(self.inputs[k][self.pointer:end, :])
            labels.append(self.labels[k][self.pointer:end, :])
            rews.append(self.rews[k][self.pointer:end])
            if self.nobs_flag:
                nobs.append(self.nobs[k][self.pointer:end, :])
        all_obs = self.all_obs[self.pointer:end, :]
        self.pointer = end
        if self.nobs_flag:
            return inputs, labels, nobs, all_obs, rews
        else:
            return inputs, labels, all_obs, rews

    def update(self, inputs, labels, nobs, all_obs, rews, decay_rate=0.9):
        idx = np.arange(self.num_pairs)
        np.random.shuffle(idx)
        l = int(self.num_pairs * decay_rate)
        # decay
        for k in range(self.num_agents):
            self.inputs[k] = self.inputs[k][idx[:l], :]
            self.labels[k] = self.labels[k][idx[:l], :]
            if self.nobs_flag:
                self.nobs[k] = self.nobs[k][idx[:l], :]
            self.rews[k] = self.rews[k][idx[:l]]
        self.all_obs = self.all_obs[idx[:l], :]
        # update
        for k in range(self.num_agents):
            self.inputs[k] = np.concatenate([self.inputs[k], inputs[k]], axis=0)
            self.labels[k] = np.concatenate([self.labels[k], labels[k]], axis=0)
            if self.nobs_flag:
                self.nobs[k] = np.concatenate([self.nobs[k], nobs[k]], axis=0)
            self.rews[k] = np.concatenate([self.rews[k], rews[k]], axis=0)
        self.all_obs = np.concatenate([self.all_obs, all_obs], axis=0)
        self.num_pairs = len(inputs[0])
        self.init_pointer()


class MADataSet(object):
    def __init__(self, expert_path, train_fraction=0.7, ret_threshold=None, traj_limitation=np.inf, randomize=True,
                 nobs_flag=False):
        self.nobs_flag = nobs_flag
        with open(expert_path, "rb") as f:
            traj_data = pkl.load(f)
        num_agents = len(traj_data[0]["ob"])
        obs = []
        acs = []
        rets = []
        lens = []
        rews = []
        obs_next = []

        all_obs = []
        for k in range(num_agents):
            obs.append([])
            acs.append([])
            rews.append([])
            rets.append([])
            obs_next.append([])

        np.random.shuffle(traj_data)

        for traj in tqdm(traj_data):
            if len(lens) >= traj_limitation:
                break
            for k in range(num_agents):
                obs[k].append(traj["ob"][k])
                acs[k].append(traj["ac"][k])
                rews[k].append(traj["rew"][k])
                rets[k].append(traj["ep_ret"][k])
            lens.append(len(traj["ob"][0]))
            all_obs.append(traj["all_ob"])
        print("observation shape:", len(obs), len(obs[0]), len(obs[0][0]), len(obs[0][0][0]))
        print("action shape:", len(acs), len(acs[0]), len(acs[0][0]), len(acs[0][0][0]))
        print("reward shape:", len(rews), len(rews[0]), len(rews[0][0]))
        print("return shape:", len(rets), len(rets[0]))
        print("all observation shape:", len(all_obs), len(all_obs[0]), len(all_obs[0][0]))
        self.num_traj = len(rets[0])
        self.avg_ret = np.sum(rets, axis=1) / len(rets[0])
        self.avg_len = sum(lens) / len(lens)
        self.rets = np.array(rets)
        self.lens = np.array(lens)
        self.obs = obs
        self.acs = acs
        self.rews = rews

        for k in range(num_agents):
            self.obs[k] = np.concatenate(self.obs[k])
            self.acs[k] = np.concatenate(self.acs[k])
            self.rews[k] = np.concatenate(self.rews[k])
        self.all_obs = np.concatenate(all_obs)

        # get next observation
        for k in range(num_agents):
            nobs = self.obs[k].copy()
            nobs[:-1] = self.obs[k][1:]
            nobs[-1] = self.obs[k][0]
            obs_next[k] = nobs
        self.obs_next = obs_next

        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)

        assert len(self.obs[0]) == len(self.acs[0])
        self.num_transition = len(self.obs[0])
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.obs_next, self.all_obs, self.rews, self.randomize, num_agents,
                         nobs_flag=self.nobs_flag)
        # for behavior cloning
        self.train_set = Dset(self.obs, self.acs, self.obs_next, self.all_obs, self.rews, self.randomize, num_agents,
                              nobs_flag=self.nobs_flag)
        self.val_set = Dset(self.obs, self.acs, self.obs_next, self.all_obs, self.rews, self.randomize, num_agents,
                            nobs_flag=self.nobs_flag)
        self.log_info()

    def log_info(self):
        logger.log("Total trajectories: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average episode length: %f" % self.avg_len)
        logger.log("Average returns:", str(self.avg_ret))

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path, ret_threshold, traj_limitation):
    dset = MADataSet(expert_path, ret_threshold=ret_threshold, traj_limitation=traj_limitation)
    a, b, c, d = dset.get_next_batch(64)
    print(a[0].shape, b[0].shape, c.shape, d[0].shape)
    # dset.plot()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str,
                        default="/Users/LantaoYu/PycharmProjects/multiagent-irl/models/mack-simple_tag-checkpoint20000-20tra.pkl")
    parser.add_argument("--ret_threshold", type=float, default=-9.1)
    parser.add_argument("--traj_limitation", type=int, default=200)
    args = parser.parse_args()
    test(args.expert_path, args.ret_threshold, args.traj_limitation)
