import logging
import json
import click
from datetime import datetime
from itertools import count
from functools import partial

import gym
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
from evolution.map_elites import MapElites
from models.dqn import DQN
from models.gvg_utils import get_screen
from models.gvg_utils import get_screen
from models.replay_memory import ReplayMemory, Transition
from models.train_dqn import evaluate_net
from models.caching_environment_maker import CachingEnvironmentMaker, GVGAI_BAM4D, GVGAI_RUBEN
from environment_utils.utils import get_run_file_name, get_run_name, find_device
import threading
logging.basicConfig(filename=get_run_file_name(),level=logging.INFO,format='[%(levelname)s] (%(threadName)-10s) %(message)s',)


SCORE_ALL = 'score_all'
SCORE_WINNING = 'score_winning'
SCORE_LOSING = 'score_losing'

def get_initial_policy_net(level='gvgai-zelda-lvl0-v0', LINEAR_INPUT_SCALAR=8,
                           KERNEL=5, env_maker=None):
    if env_maker:
        env = env_maker(level)
    else:
        import gym_gvgai
        env = gym.make(level)

    device = find_device()
    init_screen = get_screen(env, device)

    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n

    init_model = [screen_height, screen_width, LINEAR_INPUT_SCALAR, KERNEL, n_actions]
    policy_net = DQN(*init_model).to(device)
    return policy_net, init_model

def combine_scores(scores, score, win, mode):
    if mode == SCORE_ALL:
        scores += score
    elif mode == SCORE_WINNING:
        if win == 1:
            scores += score
    elif mode == SCORE_LOSING:
        if win == 0:
            scores += score
    return scores


def fitness_feature_fn(score_strategy, stop_after, game, run_name, policy_net, env_maker):
    """
    Calculate fitess and feature descriptor simultaneously
    """
    scores = 0
    wins = []
    for lvl in range(5):
        score, win = evaluate_net(policy_net,
                                  game_level=f'{game}-lvl{lvl}-v0',
                                  stop_after=stop_after,
                                  env_maker=env_maker)
        scores = combine_scores(scores, score, win, score_strategy)
        wins.append(win)

    fitness = scores
    feature_descriptor = '-'.join([str(i) for i in wins])
    return fitness, feature_descriptor


def validate_args(score_strategy,):
    if score_strategy in [SCORE_WINNING, SCORE_ALL, SCORE_LOSING]:
        return
    raise Exception('Invalid Run Arguments')



@click.command()
@click.option('--num_iter', default=2000, help='Number of module iters over which to evaluate for each algorithm.')
@click.option('--score_strategy', default=SCORE_ALL, help='Scoring strategy for algorithm')
@click.option('--game', default='gvgai-zelda', help='Which game to run')
@click.option('--stop_after', default=None, help='Number of iterations after which to stop evaluating the agent')
@click.option('--save_model', default=False, help='Whether to save the final model')
@click.option('--gvgai_version', default=GVGAI_BAM4D, help='Which version of the gvgai library to run, GVGAI_BAM4D or GVGAI_RUBEN')
@click.option('--num_threads', default=1, help='Number of multithreading threads to run for evaluating agents')
@click.option('--log_level', default='INFO', help='Logging level. DEBUG for all log statements')
@click.option('--max_age',default = 750,help = 'Maximum number of iterations elite stored in map')
@click.option('--is_mortality', is_flag=True, help = 'Turn mortality on or off for elites')
@click.option('--is_crossover', is_flag=True, help = 'Turn crossover on or off for generating new models')
@click.option('--crossover_possibility', default = 0.5, help = 'Turn crossover on or off for generating new models')
@click.option('--mutate_possibility', default = 0.95, help = 'Turn mutate on or off for generating new models')
def run(num_iter, score_strategy, game, stop_after, save_model, gvgai_version, num_threads, log_level, max_age,is_mortality,is_crossover,mutate_possibility,crossover_possibility):
    validate_args(score_strategy)

    run_name = f'{game}-iter-{num_iter}-strat-{score_strategy}-stop-after-{stop_after}'

    logging.basicConfig(filename=run_name + '.log', level=logging.INFO if log_level == 'INFO' else logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('Beginning initial map elites run')
    logging.info('Run file %s', run_name)
    print('logging setup')

    EnvMaker = CachingEnvironmentMaker(version=gvgai_version)

    bound_fitness_feature = partial(fitness_feature_fn, score_strategy, stop_after, game, run_name)
    init_level = f'{game}-lvl0-v0'
    policy_net, init_model = get_initial_policy_net(level=init_level, env_maker=EnvMaker)


    init_iter = 1
    #mutate_possibility = 0.7
    #crossover_possibility = 0.5
    map_e = MapElites(policy_net,
                      init_model,
                      init_iter,
                      num_iter,
                      is_crossover,
                      mutate_possibility,
                      crossover_possibility,
                      is_mortality,
                      max_age,
                      fitness_feature=bound_fitness_feature,
                      gvgai_version=gvgai_version)
    performances, solutions = map_e.run(num_threads)
    logging.info('Finished performances')
    logging.info('Final performances:')
    logging.info(str(performances))
    if save_model:
        logging.info('Saving pytorch models...')
        for agent_name, model_dict in solutions.items():
            torch.save(model_dict, 'torch_model_{}_{}'.format(run_name, agent_name))

if __name__ == '__main__':
    run()
