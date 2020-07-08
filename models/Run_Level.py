import math
import random
from itertools import count
from datetime import datetime

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from IPython import display

from models.dqn import DQN
from models.gvg_utils import get_screen
from environment_utils.utils import get_run_file_name, find_device
from models.caching_environment_maker import CachingEnvironmentMaker, GVGAI_BAM4D
import logging

from models.train_dqn import select_action
import gvgai

num_levels = 1 
game = 'gvgai-dzelda'
lvl = 7


env = gym.make(f'{game}-lvl{lvl}-v0')
env.reset()

device = find_device()
init_screen = get_screen(env, device)
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n
LINEAR_INPUT_SCALAR = 8
KERNEL = 5
init_model = [screen_height, screen_width, LINEAR_INPUT_SCALAR, KERNEL, n_actions]
win_factor = 100
model = DQN(*init_model)
model.load_state_dict(torch.load('saved_models/torch_model_0-1-1-1-1-1'))

current_screen = get_screen(env, device)
state = current_screen

stop_after = 1000

sum_score = 0
won = 0
key_found = 0

for lvl in range(7,8):
    level_name = f'{game}-lvl{lvl}-v0'
    print(level_name)
    env = gym.make(level_name)
    
    print('Level Started')

    for t in range(stop_after):
        print(t)
        fn = "saved_images/Steps-" + str(t) + ".png"
        plt.imsave(fn,current_screen.cpu().squeeze(0).permute(1, 2, 0).numpy())
        action,expected = select_action(state, model, n_actions)
        obs, reward, done, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        #print(obs.shape)
        #print(action)
        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(env, device)
        print("Done: ", done)
        print(reward)
        print(info)
        if not done:
            next_state = current_screen
        else:
            next_state = None

        if info['winner'] == "PLAYER_WINS" or info['winner'] == 3:
          sum_score += reward*win_factor
        else:
          sum_score += reward
        #if t % 200 == 0:
            #logging.debug('Time: {}, Reward: {}, Total Score: {}'.format(t, reward,  sum_score))

        if reward == 1 and done != True:
            key_found = 1
        # Move to the next state
        state = next_state
        if done or (stop_after and t >= int(stop_after)):
            if info['winner'] == "PLAYER_WINS" or info['winner'] == 3:
                won = 1
                #logging.debug('WIN')
                #logging.debug("Score: {}, won: {}".format(sum_score.item(), won))
            elif info['winner'] == "PLAYER_LOSES" or info['winner'] == 2:
                won = 0
                print('Lose')
                #logging.debug('LOSE')
                #logging.debug("Score: {}, won: {}".format(sum_score.item(), won))
            else:
                won = 0
                #logging.debug('obs %s, done %s, info %s', str(obs), str(done), str(info))
                #logging.debug('Eval net stopped at {} steps'.format(t))
            env.reset()
            break
        #logging.debug('Completed one level eval')
if key_found == 1:
    print("KEY FOUND")
        
env.close()
            
    
    