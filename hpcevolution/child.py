import logging
import pickle
import time
from pathlib import Path

import click

from evolution.initialization_utils import get_initial_model
from evolution.run_single_host_mapelite_train import SCORE_ALL, SCORE_WINNING, SCORE_LOSING
from hpcevolution.constants import SLEEP_TIME, AVAILABLE_EXTENSION
from hpcevolution.result import Result
from models.caching_environment_maker import CachingEnvironmentMaker, GVGAI_BAM4D
from models.evaluate_model import evaluate_net


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

def fitness_feature_fn(score_strategy, stop_after, game, run_name, policy_net, env_maker,num_levels):
    """
    Calculate fitess and feature descriptor simultaneously
    """
    scores = [0,0]
    wins = []
    keys_found = 0
    #num_levels = 10 if game == 'gvgai-dzelda' else 5
    #levels = [4,5,7,8,3,9]
    levels = [5,7,8,4,9,3]
    #for lvl in range(num_levels):
    for lvl in levels[0:num_levels]:
        logging.debug('Running %s', f'{game}-lvl{lvl}-v0')
        score, win,key_found = evaluate_net(policy_net,
                                  game_level=f'{game}-lvl{lvl}-v0',
                                  stop_after=stop_after,
                                  env_maker=env_maker)
        #scores = combine_scores(scores, score, win, score_strategy)
        if lvl in objectives[0]:
            scores[0] += score

        else:
            scores[1] += score

        wins.append(win)
        wins.append(key_found)

    #fitness = [scores,keys_found]
    fitness = scores
    feature_descriptor = '-'.join([str(i) for i in wins])
    return fitness, feature_descriptor

class Child:
    def __init__(self, unique_id, gvgai_version, run_name, game):
        self.run_name = run_name
        self.run_name_with_params = f'{run_name}-{game}'
        self.id = unique_id
        logging.basicConfig(filename='../logs/{}-child-{}.log'.format(self.run_name, self.id), level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())
        self.gvgai_version = gvgai_version
        self.env_maker = CachingEnvironmentMaker(version=gvgai_version)
        self.run_name = run_name
        policy_net, init_model = get_initial_model(gvgai_version, game)
        self.model = policy_net
        self.tasks_processed = set()
        self.reset_every = 250

        #Set up directories
        parent = Path(__file__).parent
        working_dir = parent / self.run_name
        self.AVAILABLE_AGENTS_DIR = working_dir / 'available_agents'
        self.WORK_DIR = working_dir / 'work_todo'
        self.RESULTS_DIR = working_dir / 'results'
        self.AVAILABLE_AGENTS_DIR.mkdir(exist_ok=True, parents=True)
        self.WORK_DIR.mkdir(exist_ok=True, parents=True)
        self.RESULTS_DIR.mkdir(exist_ok=True, parents=True)

        # ready to go
        self.signal_available()

    def run(self):
        while True:
            logging.info('processed so far %d', len(self.tasks_processed))
            if len(self.tasks_processed) % self.reset_every == 0:
                self.reset_environments()
            self.signal_unavailable()
            task_file = self.parse_received_task()
            if task_file:
                task, file = task_file
                logging.info('task found, starting task')
                result = self.run_task(task)
                if result:
                    logging.info('Writing result')
                    self.write_result(result)
                logging.info('removing task file %s', str(file))
                file.unlink()
                logging.info('exists {}'.format(str(file.exists())))
                logging.info('finished task')
            self.signal_available()
            logging.info('Sleeping %d', SLEEP_TIME)
            time.sleep(SLEEP_TIME)

    def parse_received_task(self):
        # LOAD NN FROM DISK
        path = self.WORK_DIR
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
        try:
            success = self.model.load_state_dict(task.model)
        except Exception as e:
            logging.info('ERROR loading model. Skipping task. Error: %s', str(e))
            return
        if success:
            try:
                fitness, feature = fitness_feature_fn(task.score_strategy, task.stop_after, task.game,
                                                                self.run_name, self.model, self.env_maker, task.num_levels,task.objectives)
                result = Result(task.run_name, task.model, feature, fitness, num_levels)
                return result
            except Exception as e:
                logging.info('ERROR running task. Error: %s', str(e))
                self.reset_environments()

    def reset_environments(self):
        del self.env_maker
        logging.info('Resetting env maker')
        time.sleep(1)
        self.env_maker = CachingEnvironmentMaker(version=self.gvgai_version)
        logging.info('...reset finished')


    def write_result(self, result):
        # MAP AND SCORE
        path = self.RESULTS_DIR / self.id
        path = path.with_suffix('.pkl')
        pickle.dump(result, path.open('wb'))
        logging.info('result written')
        return

    def signal_available(self):
        # WRITE FILE "CHILD 1 AVAILABLE"
        fn = self.AVAILABLE_AGENTS_DIR / self.id
        fn = fn.with_suffix(AVAILABLE_EXTENSION)
        fn.touch()
        return

    def signal_unavailable(self):
        # DELETE FILE "CHILD 1 AVAILABLE"
        fn = self.AVAILABLE_AGENTS_DIR / self.id
        fn = fn.with_suffix(AVAILABLE_EXTENSION)
        if fn.is_file():
            fn.unlink()
        return

@click.command()
@click.option('--unique_id', default='3', help='child id')
@click.option('--gvgai_version', default=GVGAI_BAM4D, help='Which version of the gvgai library to run, GVGAI_BAM4D or GVGAI_RUBEN')
@click.option('--run_name', default='default_run_name', help='uses directory found')
@click.option('--game', default='gvgai-dzelda', help='Which game to run')
def run(unique_id, gvgai_version, run_name, game):
    Child(unique_id, gvgai_version, run_name, game).run()

if __name__ == '__main__':
    run()
