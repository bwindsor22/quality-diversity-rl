import logging
import time
import re
import numpy as np
import torch
from itertools import islice
from pathlib import Path
from models.evaluate_model import evaluate_net
from models.evaluate_model import select_action_without_random
from evolution.run_single_host_mapelite_train import SCORE_ALL, SCORE_WINNING, SCORE_LOSING
from models.caching_environment_maker import CachingEnvironmentMaker


saves_numpy = Path(__file__).parent.parent / 'saves_numpy'


class CascadingFitnessEvaluator:
    def __init__(self, gvgai_version=None):
        self.gvgai_version = gvgai_version
        self.env_maker = CachingEnvironmentMaker(version=gvgai_version)
        self.real_world_emphasis = 1000
        self.num_cached_evals = 10
        self.num_disk_evals   = 100
        self.threshold_cached = 0.05
        self.threshold_disk   = 0.01
        self.cached_dp = self.load_cached_saved_evals()

    def run_task(self, run_name, task, model):

        all_cached_score, all_cached_correct = self.eval_cached(model)
        percent_correct = all_cached_correct / self.num_cached_evals
        if percent_correct < self.threshold_cached:
            logging.info('eval finished on cached, %f, %d', percent_correct, all_cached_score)
            return all_cached_score, 'cached'

        all_disk_score, all_disk_correct = self.eval_cached(model)
        percent_correct = all_disk_correct / (self.num_disk_evals - self.num_cached_evals)
        cached_disk_score = all_cached_score + all_disk_score + BIG_SUCCESS
        if percent_correct < self.threshold_disk:
            logging.info('eval finished on disk, %f, %d', percent_correct, cached_disk_score)
            return all_cached_score + all_disk_score, 'disk'

        fitness, feature = game_fitness_feature_fn(task.score_strategy, task.stop_after, task.game,
                                                   run_name, model, self.env_maker)
        final_score = fitness * self.real_world_emphasis + cached_disk_score
        logging.info('eval finished on real game %d = %d * %d, disk: %d, cached: %d', final_score, fitness, self.real_world_emphasis, all_disk_score, all_cached_score)
        return final_score, feature


    def eval_cached(self, model):
        all_score = 0
        all_correct = 0
        for params, screen in self.cached_dp:
            score, correct = score_action_on_screen(screen, model, params)
            all_score += score
            all_correct += correct
        return all_score, all_correct

    def eval_disk(self, model):
        all_score = 0
        all_correct = 0
        files = saves_numpy.glob('*.npy')
        for file_ in islice(files, self.num_cached_evals, self.num_disk_evals):
            params = self.parse_file_name(file_.stem)
            screen = np.load(str(file_))
            score, correct = score_action_on_screen(screen, model, params)
            all_score += score
            all_correct += correct
        return all_score, all_correct


    def load_cached_saved_evals(self):
        files = saves_numpy.glob('*.npy')
        file_cache = []
        for file_ in islice(files, self.num_cached_evals):
            file_cache.append(
                (self.parse_file_name(file_.stem), np.load(str(file_))))
        return file_cache

    def parse_file_name(self, file_name):
        name_parts = re.split(r'(c_|_crit_|_rew_|_act_)', file_name)
        return {
            'count': name_parts[2],
            'crit': name_parts[4],
            'rew': int(name_parts[6]),
            'act': int(name_parts[8])
        }

    def reset_environments(self):
        del self.env_maker
        logging.info('Resetting env maker')
        time.sleep(1)
        self.env_maker = CachingEnvironmentMaker(version=self.gvgai_version)
        logging.info('...reset finished')




BIG_SUCCESS = 1000
BIG_FAILURE = -1
def score_action_on_screen(screen, model, params):
    """
    :return: tuple: score, right decision
    """
    model_action = select_action_without_random(torch.tensor(screen), model)
    dp_action, dp_type, dp_reward = params['act'], params['crit'], params['rew']
    assert isinstance(model_action, int)
    assert isinstance(dp_reward, int)
    assert isinstance(dp_action, int)
    print('dp_type', dp_type, 'dp_action', dp_action, 'model_action', model_action, 'dp_reward', dp_reward)
    if dp_type == 'win':
        if model_action == dp_action:
            return BIG_SUCCESS, 1
        else:
            return 0, 0
    elif dp_type == 'loser':
        if model_action == dp_action:
            return BIG_FAILURE, 0
        else:
            return 0, 0
    elif dp_type == 'no':
        if model_action == dp_action:
            return dp_reward, 1
        else:
            return 0, 0
    else:
        raise RuntimeError(f'Unknown decision point {dp_type}, {dp_action}, {dp_reward}')



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



def game_fitness_feature_fn(score_strategy, stop_after, game, run_name, policy_net, env_maker):
    """
    Calculate fitess and feature descriptor simultaneously
    """
    scores = 0
    wins = []
    num_levels = 10 if game == 'gvgai-dzelda' else 5
    for lvl in range(num_levels):
        logging.debug('Running %s', f'{game}-lvl{lvl}-v0')
        score, win = evaluate_net(policy_net,
                                  game_level=f'{game}-lvl{lvl}-v0',
                                  stop_after=stop_after,
                                  env_maker=env_maker)
        scores = combine_scores(scores, score, win, score_strategy)
        wins.append(win)

    fitness = scores
    feature_descriptor = '-'.join([str(i) for i in wins])
    return fitness, feature_descriptor
