import math
import random
from itertools import count
from datetime import datetime
import uuid

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
import pickle
from pathlib import Path

steps_done = 0

SAVE_DIR = Path(__file__).parent.parent / 'saves'
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

device = find_device()

win_factor = 100
save_every = 10000

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
                 stop_after=None,
                 ):

    logging.info('making level %s', game_level)

    if env_maker:
        env = env_maker(game_level)
    else:
        import gym_gvgai
        env = gym.make(game_level)
        env.reset()

    global steps_done
    steps_done = 0

    init_screen = get_screen(env, device)
    _, _, screen_height, screen_width = init_screen.shape

    n_actions = env.action_space.n

    current_screen = get_screen(env, device)
    state = current_screen
    sum_score = 0
    won = 0
    run_id = str(uuid.uuid4())


    for t in count():
        action = select_action(state, policy_net, n_actions)

        if (torch.isnan(action).any() or torch.isinf(action).any() or action == None):
            sum_score += -1
            logging.debug("Invalid Action Output By Model")
            logging.debug("Score: {}, won: {}".format(sum_score.item(), won))
            return sum_score,won

        obs, reward_raw, done, info = env.step(action.item())
        reward = torch.tensor([reward_raw], device=device)
        is_winner = info['winner'] == "PLAYER_WINS" or info['winner'] == 3
        is_loser = info['winner'] == "PLAYER_LOSES" or info['winner'] == 2

        numpy_save(SAVE_DIR, run_id, game_level, state, action, t, reward_raw, is_winner, is_loser)

        if is_winner:
          sum_score += reward*win_factor
        else:
          sum_score += reward
        if t % 1000 == 0:
            logging.info('Time: {}, Reward: {}, Total Score: {}'.format(t, reward,  sum_score))

        # Observe new state
        state = get_screen(env, device)

        # Move to the next state
        if done or (stop_after and t >= int(stop_after)):
            if is_winner:
                won = 1
                logging.debug('WIN')
                logging.debug("Score: {}, won: {}".format(sum_score.item(), won))
            elif is_loser:
                won = 0
                logging.debug('LOSE')
                logging.debug("Score: {}, won: {}".format(sum_score.item(), won))
            else:
                won = 0
                logging.debug('obs %s, done %s, info %s', str(obs), str(done), str(info))
                logging.debug('Eval net stopped at {} steps'.format(t))
            break

    logging.info('Completed one level eval')
    numpy_save(SAVE_DIR, run_id, game_level, current_screen, action, t, reward_raw, is_winner, is_loser)
    env.close()

    return sum_score, won, t

def numpy_save(SAVE_DIR, run_id, game_level, current_screen, action, t, reward_raw, is_winner, is_loser):
    screen_numpy = current_screen.numpy()
    action_val = action.item()
    win_val = 'win' if is_winner else 'lose' if is_loser else '__'
    file_name = SAVE_DIR / f'{run_id}_{game_level}_step_{t}_act_{action_val}_reward_{reward_raw}_win_{win_val}.npy'
    with open(str(file_name), 'wb') as f:
        np.save(f, screen_numpy)

def history_dict(current_screen, action, reward_raw, info, is_winner, is_loser):
    if 'actions' in info:
        del info['actions']
    return {
        'current_screen': current_screen.numpy(),
        'action': action.item(),
        'reward': reward_raw,
        'info': info,
        'critical': 'winner' if is_winner else 'loser' if is_loser else 'no'
    }

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
