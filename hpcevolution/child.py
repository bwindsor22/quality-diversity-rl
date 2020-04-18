import time

import click
import pickle
import os
import logging

import torch

from evolution.initialization_utils import get_initial_model
from evolution.run_single_host_mapelite_train import SCORE_ALL, SCORE_WINNING, SCORE_LOSING
from hpcevolution.result import Result
from models.caching_environment_maker import CachingEnvironmentMaker, GVGAI_RUBEN
from models.dqn import DQN
from models.evaluate_model import evaluate_net
from hpcevolution.constants import SLEEP_TIME, AVAILABLE_AGENTS_DIR_PATHLIB, AVAILABLE_EXTENSION, RESULTS_DIR_PATHLIB, \
    WORK_DIR_PATHLIB


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

class Child:
    def __init__(self, unique_id, gvgai_version, run_name, game):
        self.run_name = f'{game}'
        self.id = unique_id
        logging.basicConfig(filename='../logs/{}-child-{}.log'.format(self.run_name, self.id), level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())
        self.is_available = False
        self.env_maker = CachingEnvironmentMaker(version=gvgai_version)
        self.run_name = run_name
        policy_net, init_model = get_initial_model(gvgai_version, game)
        self.model = policy_net
        self.signal_available()
        self.tasks_processed = set()

    def run(self):
        while True:
            logging.info('processed so far')
            logging.info(str(self.tasks_processed))
            self.signal_unavailable()
            task_file = self.parse_received_task()
            if task_file:
                task, file = task_file
                logging.info('task found, starting task')
                feature, fitness = self.run_task(task)
                res = Result(task.run_name,task.model,feature,fitness)
                logging.info('Writing result')
                self.write_result(res)
                logging.info('removing task file')
                file.unlink()
                logging.info('exists {}'.format(str(file.exists())))
                logging.info('finished task')
            self.signal_available()
            logging.info('Sleeping %d', SLEEP_TIME)
            time.sleep(SLEEP_TIME)

    def parse_received_task(self):
        # LOAD NN FROM DISK
        path = WORK_DIR_PATHLIB
        files = path.glob('*.pkl')
        for file in files:
            if 'child-{}'.format(self.id) in file.stem and file.is_file():
                file_name = file.stem
                if file_name not in self.tasks_processed:
                    logging.info('starting %s', file_name)
                    self.tasks_processed.add(file_name)
                    task = pickle.load(file.open('rb'))
                    return task, file
        return None

    def run_task(self, task):
        #RUN evaluate model
        success = self.model.load_state_dict(task.model)
        if success:
            fitness, feature = fitness_feature_fn(task.score_strategy, task.stop_after, task.game,
                                                            self.run_name, self.model, self.env_maker)
            return feature, fitness
        

    def write_result(self, result):
        # MAP AND SCORE
        path = RESULTS_DIR_PATHLIB / self.id
        path = path.with_suffix('.pkl')
        pickle.dump(result, path.open('wb'))
        return

    def signal_available(self):
        # WRITE FILE "CHILD 1 AVAILABLE"
        fn = AVAILABLE_AGENTS_DIR_PATHLIB / self.id
        fn = fn.with_suffix(AVAILABLE_EXTENSION)
        fn.touch()
        return

    def signal_unavailable(self):
        # DELETE FILE "CHILD 1 AVAILABLE"
        fn = AVAILABLE_AGENTS_DIR_PATHLIB / self.id
        fn = fn.with_suffix(AVAILABLE_EXTENSION)
        if fn.is_file():
            fn.unlink()
        return

@click.command()
@click.option('--unique_id', default='3', help='child id')
@click.option('--gvgai_version', default=GVGAI_RUBEN, help='Which version of the gvgai library to run, GVGAI_BAM4D or GVGAI_RUBEN')
@click.option('--run_name', default='1', help='TODO')
@click.option('--game', default='gvgai-zelda', help='Which game to run')
def run(unique_id, gvgai_version, run_name, game):
    Child(unique_id, gvgai_version, run_name, game).run()

if __name__ == '__main__':
    run()