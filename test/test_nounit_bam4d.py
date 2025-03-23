import gym
import gvgai

import logging
print('running test')
print('running make')
env = gym.make('gvgai-aliens-lvl0-v0')
print('running reset')
env.reset()
sum = 0
for t in range(2000):
    print('now doing a sample')
    action_id = env.action_space.sample()
    print('stepping', action_id)
    state, reward, isOver, debug = env.step(action_id)
    sum = sum + reward
    print('action', action_id, 'time', t, 'reward', reward, 'sum', sum)
    logging.info('this is some info at time %d', t)
