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

from evolution.initialization_utils import get_initial_model
from evolution.map_elites import MapElites
from evolution.singe_host_map_elites import SingleHostMapElites
from models.dqn import DQN
from models.gvg_utils import get_screen
from models.gvg_utils import get_screen
from models.replay_memory import ReplayMemory, Transition
from models.evaluate_model import evaluate_net
from models.caching_environment_maker import CachingEnvironmentMaker, GVGAI_BAM4D, GVGAI_RUBEN
from environment_utils.utils import get_run_file_name, get_run_name, find_device


SCORE_ALL = 'score_all'
SCORE_WINNING = 'score_winning'
SCORE_LOSING = 'score_losing'


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
@click.option('--mutate_possibility', default = 0.7, help = 'Turn mutate on or off for generating new models')
@click.option('--mepgd_possibility', default = 0.7, help = 'Turn mutate on or off for generating new models')
@click.option('--is_mepgd', is_flag=True, help = 'Turn crossover on or off for generating new models')
@click.option('--cmame', is_flag=True, help='run CMA-ME')
def run(num_iter, score_strategy, game, stop_after, save_model, gvgai_version, num_threads, log_level, max_age,is_mortality,
        is_crossover,crossover_possibility,mutate_possibility,mepgd_possibility,is_mepgd, cmame):
    validate_args(score_strategy)

    run_name = f'{game}-iter-{num_iter}-strat-{score_strategy}-stop-after-{stop_after}'

    logging.basicConfig(filename=run_name + '.log', level=logging.INFO if log_level == 'INFO' else logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('Beginning initial map elites run')
    logging.info('Run file %s', run_name)
    print('logging setup')

    bound_fitness_feature = partial(fitness_feature_fn, score_strategy, stop_after, game, run_name)

    policy_net, init_model = get_initial_model(gvgai_version, game)

    init_iter = 1
    #mutate_possibility = 0.7
    #crossover_possibility = 0.5
    map_e = SingleHostMapElites(policy_net,
                      init_model,
                      init_iter,
                      num_iter,
                      is_crossover,
                      mutate_possibility,
                      crossover_possibility,
                      is_mortality,
                      max_age,
                      is_mepgd,
                      mepgd_possibility,
                      fitness_feature=bound_fitness_feature,
                      gvgai_version=gvgai_version,
                      is_cmame=cmame)
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
