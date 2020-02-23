import gym
import gym
import gym_gvgai
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from IPython import display
from datetime import datetime
from itertools import count

from evolution.map_elites import MapElites
from models.dqn import DQN
from models.gvg_utils import get_screen
from models.gvg_utils import get_screen
from models.replay_memory import ReplayMemory, Transition
from models.train_dqn import run_training_for_params


def get_initial_policy_net(level='gvgai-zelda-lvl0-v0', LINEAR_INPUT_SCALAR=8,
                           KERNEL=5):
    env = gym.make(level)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_screen = get_screen(env, device)

    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n

    init_model = [screen_height, screen_width, LINEAR_INPUT_SCALAR, KERNEL, n_actions]
    policy_net = DQN(*init_model).to(device)
    return policy_net, init_model


def run():
    def fitness_feature(policy_net):
        """
        Calculate fitess and feature descriptor simultaneously
        :param policy_net:
        :return:
        """
        scores = 0
        wins = []
        for lvl in range(5):
            score, win = run_training_for_params(policy_net, game_level='gvgai-zelda-lvl{}-v0'.format(lvl))
            scores += score
            wins.append(win)

        fitness = scores
        feature_descriptor = '-'.join(wins)
        return fitness, feature_descriptor

    policy_net, init_model = get_initial_policy_net()

    map_e = MapElites(policy_net,
                      init_model,
                      1,
                      5,
                      0.5,
                      0.7,
                      fitness_feature=fitness_feature)
    map_e.run()

if __name__ == '__main__':
    run()