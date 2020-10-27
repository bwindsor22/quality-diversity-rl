import gym
import logging
from environment_utils.zelda_env_bam4d import ZeldaEnv

GVGAI_RUBEN = 'GVGAI_RUBEN'
GVGAI_BAM4D = 'GVGAI_BAM4D'

class CachingEnvironmentMaker:
    """
    Speeds up repeat calls to env.make.
    Stores the old environment so python doesn't have to recreate and reconnect.
    """
    def __init__(self, version=GVGAI_RUBEN, make_env_attempts=100):
        if version == GVGAI_RUBEN:
            #usually runs, slower
            import gym_gvgai
        elif version == GVGAI_BAM4D:
            # harder to get running, is faster
            import gvgai

        self.make_env_attempts = make_env_attempts
        self.cache = {}

    def __call__(self, level):
        return self.make(level)

    def make(self, level):
        if level in self.cache:
            self.ensure_working_env(level)
        else:
            env = self.try_to_make(level)
            self.cache[level] = env
        logging.debug('returning level %s', level)
        return self.cache[level]

    def ensure_working_env(self, level):
        env = self.cache[level]
        try:
        # if True:
            self.test_env(env)
            logging.debug('found env in cache')
            return
        except Exception:
            logging.debug('Cached env failed, remaking env')

        self.cache[level] = self.try_to_make(level)

    def try_to_make(self, level):
        for att in range(self.make_env_attempts):
            try:
            # if True:
                env = gym.make(level,tile_observations = True)
                env = ZeldaEnv(env, crop=True, rotate=True, full=False, repava=True, shape=(84,84))
                self.test_env(env)
                return env
            except Exception as e:
                logging.debug('Failed making environment, attempt %d, err %s',
                            att, str(e))
        logging.error('Unable to make env after %d attempts', self.make_env_attempts)

    def test_env(self, env):
        env.reset()
        action_id = env.action_space.sample()
        env.step(action_id)
        env.reset()
