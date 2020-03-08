import gym
import gym_gvgai

from unittest import TestCase

class RubenTest(TestCase):
    """
    python -m unittest test/test_ruben.py
    """
    def test_ruben(self):
        env = gym.make('gvgai-aliens-lvl0-v0')

        env.reset()
        sum = 0
        for t in range(2000):
            action_id = env.action_space.sample()
            print(action_id)
            state, reward, isOver, debug = env.step(action_id)
            sum = sum + reward
            print('action', action_id, 'time', t, 'reward', reward, 'sum', sum)
