import math
import random
from itertools import count
from datetime import datetime

import gym
import gym_gvgai
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.distributions import Categorical
from IPython import display

from models.actor_critic import a2c
from models.gvg_utils import get_screen
from environment_utils.utils import get_run_file_name, find_device
import logging
logging.basicConfig(filename=get_run_file_name(),level=logging.INFO)

steps_done = 0

device = find_device()


def select_action(state, model, n_actions):
    
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    
    return action.item()


def run_training_for_params(policy_net,
                            game_level):
    logging.info('making level %s', game_level)
    env = gym.make(game_level)

    global steps_done
    steps_done = 0

    init_screen = get_screen(env, device)
    _, _, screen_height, screen_width = init_screen.shape

    n_actions = env.action_space.n

    env.reset()
    last_screen = get_screen(env, device)
    current_screen = get_screen(env, device)
    state = current_screen - last_screen
    sum_score = 0
    won = 0

    for t in count():
        action = select_action(state, policy_net, n_actions)
        
        _, reward, done, info = env.step(action)
        reward = torch.tensor([reward], device=device)


        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(env, device)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        sum_score += reward
        
        
        if t % 200 == 0:
            logging.info('Time: {}, Reward: {}, Total Score: {}'.format(t, reward,  sum_score))


        # Move to the next state
        state = next_state
        if done:
            if info['winner'] == "PLAYER_WINS":
                won = 1
                logging.info('WIN')
                logging.info("Score: {}, won: {}".format(sum_score.item(), won))
            else:
                won = 0
                logging.info('LOSE')
                logging.info("Score: {}, won: {}".format(sum_score.item(), won))
            break
        
        #if t==249:
            #break
        
    logging.info('Completed one level eval')

    env.close()
    return sum_score, won


if __name__ == '__main__':
    logging.info('running main')
    def get_initial_policy_net(LINEAR_INPUT_SCALAR=32,
                               KERNEL=11):
        env = gym.make('gvgai-zelda-lvl0-v0')
        init_screen = get_screen(env, device)

        _, _, screen_height, screen_width = init_screen.shape
        n_actions = env.action_space.n

        init_model = [screen_height, screen_width, LINEAR_INPUT_SCALAR, KERNEL, n_actions]
        policy_net = a2c(*init_model).to(device)
        return policy_net, init_model
    net, model = get_initial_policy_net()

    run_training_for_params(net, 'gvgai-zelda-lvl0-v0')
