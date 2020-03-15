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
import logging

steps_done = 0

device = find_device()


def select_action(state, policy_net, n_actions,
                  EPS_START=0.05,
                  EPS_END=0.05,
                  EPS_DECAY=200,
                  ):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def evaluate_net(policy_net,
                 game_level,
                 env_maker=None,
                 stop_after=None
                 ):

    logging.debug('making level %s', game_level)
    
    if env_maker:
        env = env_maker.resource.make(game_level)
        is_Tilemap = env_maker.resource.is_Tilemap
        #print(is_Tilemap)
    else:
        import gym_gvgai
        env = gym.make(game_level)
        env.reset()
        is_Tilemap = False

    global steps_done
    steps_done = 0
    #print("Env maker is_Tilemap: ", is_Tilemap)
    if is_Tilemap == False:
        init_screen = get_screen(env, device)
        _, _, screen_height, screen_width = init_screen.shape
    else:
        _,init_screen = env.reset()
        #print(init_screen)
        screen_height, screen_width,depth = init_screen.shape

    n_actions = env.action_space.n
    
    if is_Tilemap == False:
        last_screen = get_screen(env, device)
        current_screen = get_screen(env, device)
    else:
        _,last_screen = env.reset()
        #print("last_screen")
        #print(last_screen.shape)
        current_screen = np.copy(last_screen)
    
    state = current_screen - last_screen
    if is_Tilemap == True:
        state = torch.from_numpy(state).float().to(device)
        #print(state[0])
        state = state.view(-1,depth,screen_width,screen_height)
    #
    sum_score = 0
    won = 0

    for t in count():
        
        action = select_action(state, policy_net, n_actions)
        
        observation, reward, done, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        
        if is_Tilemap == False:
            current_screen = get_screen(env, device)
        else:
            current_screen = observation[1]
        
        
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        sum_score += reward
        if t % 200 == 0:
            logging.debug('Time: {}, Reward: {}, Total Score: {}'.format(t, reward,  sum_score))


        # Move to the next state
        #print(observation[1].shape)
        state = next_state
        if(done == False and is_Tilemap == True):
            state = torch.from_numpy(state).float().to(device)
            state = state.view(-1,depth,screen_width,screen_height)        
        if done or (stop_after and t >= int(stop_after)):
            if info['winner'] == "PLAYER_WINS":
                won = 1
                logging.debug('WIN')
                logging.debug("Score: {}, won: {}".format(sum_score.item(), won))
            elif info['winner'] == "PLAYER_LOSES":
                won = 0
                logging.debug('LOSE')
                logging.debug("Score: {}, won: {}".format(sum_score.item(), won))
            else:
                won = 0
                logging.debug('Eval net stopped at {} steps'.format(t))
            break

    logging.debug('Completed one level eval')

    env.close()
    return sum_score, won


if __name__ == '__main__':
    """
    Test function for running
    """
    import gym_gvgai
    logging.info('running main')
    def get_initial_policy_net(LINEAR_INPUT_SCALAR=8,
                               KERNEL=5):
        env = gym.make('gvgai-zelda-lvl0-v0')
        init_screen = get_screen(env, device)

        _, _, screen_height, screen_width = init_screen.shape
        n_actions = env.action_space.n

        init_model = [screen_height, screen_width, LINEAR_INPUT_SCALAR, KERNEL, n_actions]
        policy_net = DQN(*init_model).to(device)
        return policy_net, init_model
    net, model = get_initial_policy_net()

    evaluate_net(net, 'gvgai-zelda-lvl0-v0')
