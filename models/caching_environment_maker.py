import gym
import gym_gvgai

class CachingEnvironmentMaker:
    """
    Speeds up repeat calls to env.make.
    Stores the old environment so python doesn't have to recreate and reconnect.
    """
    def __init__(self):
        self.cache = {}

    def make(self, level):
        if level in self.cache:
            env = self.cache[level]
            env.reset()
        else:
            env = gym.make(level)
            self.cache[level] = env
        return env