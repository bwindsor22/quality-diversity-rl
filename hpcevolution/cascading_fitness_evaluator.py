import logging
import time
import re
import numpy as np
import torch
from datetime import datetime
from itertools import islice
from pathlib import Path
from batch_data_prep.file_rw_utils import parse_name
from models.evaluate_model import evaluate_net
from models.evaluate_model import select_action_without_random
from evolution.run_single_host_mapelite_train import SCORE_ALL, SCORE_WINNING, SCORE_LOSING
from models.caching_environment_maker import CachingEnvironmentMaker

saves_numpy = Path(__file__).parent.parent / 'saves_numpy'


class CascadingFitnessEvaluator:
    def __init__(self, gvgai_version=None):
        self.gvgai_version = gvgai_version
        self.env_maker = CachingEnvironmentMaker(version=gvgai_version)
        self.default_count = 1000

        self.attack_to_score = '*attW*.npy' #
        self.attack_to_lose = '*attL*.npy' #
        self.reward_1 = '*keyget_*.npy' #
        self.do_win = '*winseq_*.npy' #

        self.threshold_score = 31000


    def run_task(self, run_name, task, model):
        total_score = 0

        logging.info('beginning attack to score')
        start = datetime.now()
        def eval_function(act, record):
            return int(act) == int(record)
        att_score = self.eval_model(model, self.attack_to_score, eval_function, self.default_count)
        total_score += att_score
        logging.info('Finished attack to score with %d score, total: %d,  in %s', att_score, total_score, str(datetime.now() - start))


        logging.info('beginning attack to lose')
        start = datetime.now()
        def eval_function(act, record):
            return int(act) != int(record)
        no_att_score = self.eval_model(model, self.attack_to_lose, eval_function, self.default_count)
        total_score += no_att_score
        logging.info('Finished attack to lose with %d score, total: %d,  in %s', no_att_score, total_score, str(datetime.now() - start))

        if not (att_score > 0 and no_att_score > 0):
            return total_score, 'attack_no_attack'

        logging.info('beginning reward_1')
        start = datetime.now()
        def eval_function(act, record):
            return int(act) == int(record)
        rew_1_score = self.eval_model(model, self.reward_1, eval_function, self.default_count)
        total_score += rew_1_score
        logging.info('Finished reward_1 with %d score, total: %d,  in %s', rew_1_score, total_score, str(datetime.now() - start))

        logging.info('beginning do win')
        start = datetime.now()
        def eval_function(act, record):
            return int(act) == int(record)
        win_score = self.eval_model(model, self.do_win, eval_function, self.default_count)
        total_score += win_score
        logging.info('Finished do win with %d score, total: %d,  in %s', win_score, total_score, str(datetime.now() - start))

        if total_score < self.threshold_score or not (rew_1_score > 0 and win_score > 0):
            return total_score, 'rew_1_win'


        logging.info('beginning game eval')
        start = datetime.now()
        fitness, feature = game_fitness_feature_fn(task.score_strategy, task.stop_after, task.game,
                                                   run_name, model, self.env_maker)
        fitness = fitness.item() if torch.is_tensor(fitness) else fitness
        total_score = total_score + fitness * 100
        logging.info('eval finished on real game, total: %d, fitness: %s, feature: %s, time: %s', total_score, str(fitness), str(feature), str(datetime.now() - start))
        return total_score, feature


    def eval_model(self, model, dir, eval_function, eval_count):
        score = 0
        i = 0
        # for i, file in enumerate(dir.glob('*.npy')):
        for file in saves_numpy.glob(dir):
            parts = parse_name(file.stem)
            if i >= eval_count:
                return score
            screen = np.load(str(file))
            model_action = select_action_without_random(torch.tensor(screen), model)
            score += eval_function(model_action, parts['act'])
            i += 1
        logging.info('dir %s, i %d', str(dir), i)
        return score


    def reset_environments(self):
        del self.env_maker
        logging.info('Resetting env maker')
        time.sleep(1)
        self.env_maker = CachingEnvironmentMaker(version=self.gvgai_version)
        logging.info('...reset finished')



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


