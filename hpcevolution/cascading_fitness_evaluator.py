import logging
import time
import re
import numpy as np
import torch
from datetime import datetime
from itertools import islice
from pathlib import Path
from models.evaluate_model import evaluate_net
from models.evaluate_model import select_action_without_random
from evolution.run_single_host_mapelite_train import SCORE_ALL, SCORE_WINNING, SCORE_LOSING
from models.caching_environment_maker import CachingEnvironmentMaker

saves_formatted = Path(__file__).parent.parent / 'saves_formatted'


class CascadingFitnessEvaluator:
    def __init__(self, gvgai_version=None):
        self.gvgai_version = gvgai_version
        self.env_maker = CachingEnvironmentMaker(version=gvgai_version)
        self.default_count = 100000

        self.attack_to_score_dir = saves_formatted / 'other' / '2.0' / '1'
        self.attack_to_lose_dir = saves_formatted / 'lose' / '-1.0' / '1'
        self.do_not_lose_dir = saves_formatted / 'lose' / '-1.0' / 'other'


        start = datetime.now()

    def run_task(self, run_name, task, model):
        total_score = 0

        logging.info('beginning attack to score')
        start = datetime.now()
        def eval_function(act, record):
            return int(act) == 1
        att_score = self.eval_model(model, self.attack_to_score_dir, eval_function, self.default_count)
        total_score += att_score
        logging.info('Finished attack to score with %d score, total: %d,  in %s', att_score, total_score, str(datetime.now() - start))


        logging.info('beginning attack to lose')
        start = datetime.now()
        def eval_function(act, record):
            return int(act) != 1
        no_att_score = self.eval_model(model, self.attack_to_score_dir, eval_function, self.default_count)
        total_score += no_att_score
        logging.info('Finished attack to lose with %d score, total: %d,  in %s', no_att_score, total_score, str(datetime.now() - start))

        if not (att_score > 0 and no_att_score > 0):
            return total_score, 'training'

        logging.info('beginning do not lose')
        start = datetime.now()
        def eval_function(act, record):
            return int(act) != int(record)
        score = self.eval_model(model, self.attack_to_score_dir, eval_function, self.default_count)
        total_score += score
        logging.info('Finished do not lose with %d score, total: %d,  in %s', score, total_score, str(datetime.now() - start))

        logging.info('beginning game eval')
        start = datetime.now()
        fitness, feature = game_fitness_feature_fn(task.score_strategy, task.stop_after, task.game,
                                                   run_name, model, self.env_maker)
        fitness = fitness.item() if torch.is_tensor(fitness) else fitness
        total_score = total_score + fitness
        logging.info('eval finished on real game, total: %d, fitness: %s, feature: %s, time: %s', total_score, str(fitness), str(feature), str(datetime.now() - start))
        return total_score, feature


    def eval_model(self, model, dir, eval_function, eval_count):
        score = 0
        for i, file in enumerate(dir.glob('*.npy')):
            if i >= eval_count:
                return score
            screen = np.load(str(file))
            parts = parse_name(file.stem)
            model_action = select_action_without_random(torch.tensor(screen), model)
            score += eval_function(model_action, parts['act'])
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


def parse_name(file_name):
    items = dict()
    parts = file_name.split('_')
    items['uuid'] = parts[0]
    items['lvl'] = parts[1]
    is_name = True
    name = ''
    for part in parts[2:]:
        if is_name:
            name = part
            is_name = False
        else:
            items[name] = part
            is_name = True
    return items
