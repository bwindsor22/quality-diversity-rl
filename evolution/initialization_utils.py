import gym

from environment_utils.utils import find_device
from models.caching_environment_maker import CachingEnvironmentMaker
from models.dqn import DQN
from models.gvg_utils import get_screen
from environment_utils.zelda_env_bam4d import ZeldaEnv


def get_initial_model(gvgai_version, game):
    EnvMaker = CachingEnvironmentMaker(version=gvgai_version)

    init_level = f'{game}-lvl0-v0'
    policy_net, init_model = get_initial_policy_net(level=init_level, env_maker=EnvMaker)
    return policy_net, init_model

def get_initial_policy_net(level='gvgai-zelda-lvl0-v0', LINEAR_INPUT_SCALAR=16,
                           KERNEL=8, env_maker=None):
    if env_maker:
        env = env_maker(level)
    else:
        import gym_gvgai
        env = gym.make(level,tile_observations = True)
        env = ZeldaEnv(env, crop=True, rotate=True, full=False, repava=True, shape=(84,84))

    device = find_device()
    init_screen = get_screen(env, device)

    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n

    init_model = [screen_height, screen_width, LINEAR_INPUT_SCALAR, KERNEL, n_actions]
    policy_net = DQN(*init_model).to(device)
    return policy_net, init_model
