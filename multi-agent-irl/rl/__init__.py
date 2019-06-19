from gym.envs.registration import register
from rl.envs.multi_walker import MultiWalkerEnv


def make_env(env_id):
    if env_id == 'walker1':
        return MultiWalkerEnv(n_walkers=1, reward_mech='local')
    elif env_id == 'walker2':
        return MultiWalkerEnv(n_walkers=2, reward_mech='local')
    elif env_id == 'walker3':
        return MultiWalkerEnv(n_walkers=3, reward_mech='local')
    elif env_id == 'walker2c':
        return MultiWalkerEnv(n_walkers=2, reward_mech='local', competitive=True)
    elif env_id == 'walker4':
        return MultiWalkerEnv(n_walkers=4, reward_mech='local')
