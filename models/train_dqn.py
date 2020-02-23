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
from models.replay_memory import ReplayMemory, Transition
from models.gvg_utils import get_screen

steps_done = 0
is_ipython = 'inline' in matplotlib.get_backend()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize = T.Compose([T.ToPILImage(),
                    T.Resize(30),
                    T.ToTensor()])

plt.ion()


def select_action(state, policy_net,
                  EPS_START=0.9,
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
                            game_level,
                            BATCH_SIZE=128,
                            GAMMA=0.999,
                            TARGET_UPDATE=10,
                            LINEAR_INPUT_SCALAR=8,
                            KERNEL=5,
                            EPISODES=100):
    env = gym.make(game_level)

    global steps_done
    steps_done = 0

    init_screen = get_screen(env, device)
    _, _, screen_height, screen_width = init_screen.shape

    print(screen_height, " ", screen_width)

    n_actions = env.action_space.n

    target_net = DQN(screen_height, screen_width, LINEAR_INPUT_SCALAR, KERNEL, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    memory = ReplayMemory(10000)
    results = []
    env.reset()
    last_screen = get_screen(env, device)
    current_screen = get_screen(env, device)
    state = current_screen - last_screen
    sum_score = 0

    for t in count():
        action = select_action(state, policy_net)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)


        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(env, device)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        sum_score += reward
        #         if t % 100 == 0:
        #             print("Time: ", t, " Reward: ", reward[0], "Score: ", sum_score[0], " Aliens Killed: ",aliens_killed)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        if done:
            if reward == 2:
                print("WIN \n" * 10)
                print("Score: ", sum_score.item(), " Aliens Killed: ", aliens_killed)
                results.append([sum_score.item(), 1])
            elif reward == -1:
                print("LOSE \n" * 10)
                print("Score: ", sum_score.item(), " Aliens Killed: ", aliens_killed)
                results.append([sum_score.item(), 0])
            break

    print('Complete')
    final_result = results[-1]

    sum_all_scores = sum(r[0] for r in results)
    env.close()
    return sum_all_scores


if __name__ == '__main__':
    def get_initial_policy_net(level='gvgai-zelda-lvl0-v0', LINEAR_INPUT_SCALAR=8,
                               KERNEL=5):
        env = gym.make(level)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _, _, screen_height, screen_width = init_screen.shape
        n_actions = env.action_space.n

        init_model = [screen_height, screen_width, LINEAR_INPUT_SCALAR, KERNEL, n_actions]
        policy_net = DQN(*init_model).to(device)
        return policy_net, init_model
    net, model = get_initial_policy_net()

    run_training_for_params(net, model)
