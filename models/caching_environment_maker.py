import gym
import logging

GVGAI_RUBEN = 'GVGAI_RUBEN'
GVGAI_BAN4D = 'GVGAI_BAN4D'

class CachingEnvironmentMaker:
    """
    Speeds up repeat calls to env.make.
    Stores the old environment so python doesn't have to recreate and reconnect.
    """
    def __init__(self, version=GVGAI_RUBEN, make_env_attempts=100):
        if version == GVGAI_RUBEN:
            #usually runs, slower
            import gym_gvgai
        elif version == GVGAI_BAN4D:
            # harder to get running, is faster
            import gvgai

        self.make_env_attempts = make_env_attempts
        self.cache = {}

    def make(self, level):
        if level in self.cache:
            self.ensure_working_env(level)
        else:
            env = self.try_to_make(level)
            self.cache[level] = env
        print('returning level', level)
        return self.cache[level]

    def ensure_working_env(self, level):
        env = self.cache[level]
        try:
        # if True:
            self.test_env(env)
            logging.info('found env in cache')
            return
        except Exception:
            logging.info('Cached env failed, remaking env')

        self.cache[level] = self.try_to_make(level)

    def try_to_make(self, level):
        for att in range(self.make_env_attempts):
            try:
            # if True:
                env = gym.make(level)
                self.test_env(env)
                return env
            except Exception as e:
                logging.info('Failed making environment, attempt %d, err %s',
                            att, str(e))
        logging.error('Unable to make env after %d attempts', self.make_env_attempts)

    def test_env(self, env):
        env.reset()
        action_id = env.action_space.sample()
        env.step(action_id)
        env.reset()
