import gym
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
from rl.common.atari_wrappers import wrap_deepmind
from rl.common import set_global_seeds
import time
import numpy as np

env_id = 'SpaceInvaders'
seed = 42
nenvs = 1
np.random.seed(0)

def make_env(rank):
    def env_fn():
        env = gym.make('{}NoFrameskip-v4'.format(env_id))
        env.seed(seed + rank)
        return env
        return wrap_deepmind(env)
    return env_fn

if 1:
    from rl_algs.common.vec_env.mpi_vec_env1 import MpiVecEnv
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    nenvs = comm.Get_size()
    env = make_env(comm.Get_rank())()
    env = MpiVecEnv(env, comm)
    A = np.array([env.action_space.sample() for _ in range(env.num_envs)])*0
elif 1:
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    A = np.array([env.action_space.sample() for _ in range(env.num_envs)])*0
else:
    env = make_env(0)()
    A = env.action_space.sample()*0
    env.num_envs = 1

env.reset()

nsteps = 1000
tstart = time.time()
blah = 0
for _ in range(nsteps):
    ob,rew,done,_ = env.step(A)
    for q in (ob, rew, done):
        blah += np.sum(q)
print(blah)
totaltime = time.time() - tstart
totalframes = nsteps * env.num_envs
print('%s in %s: %s '%(totalframes, totaltime, totalframes/totaltime))

env.close()