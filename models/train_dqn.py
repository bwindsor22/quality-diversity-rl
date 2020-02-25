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
from IPython import display

from models.dqn import DQN
from models.gvg_utils import get_screen
from environment_utils.utils import get_run_file_name, find_device
import logging
logging.basicConfig(filename=get_run_file_name(),level=logging.INFO)

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
        _, reward, done, info = env.step(action.item())
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
            elif info['winner'] == "PLAYER_LOSES":
                won = 0
                logging.info('LOSE')
                logging.info("Score: {}, won: {}".format(sum_score.item(), won))
            break

    logging.info('Completed one level eval')

    env.close()
    return sum_score, won


if __name__ == '__main__':
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

    run_training_for_params(net, 'gvgai-zelda-lvl0-v0')
