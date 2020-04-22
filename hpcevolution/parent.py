import time

import torch
import pickle
from pathlib import Path
import logging

from evolution.hpc_map_elites import HPCMapElites
from evolution.initialization_utils import get_initial_model
from hpcevolution.work import Work
from hpcevolution.constants import SLEEP_TIME, AVAILABLE_AGENTS_DIR_PATHLIB, AVAILABLE_EXTENSION, RESULTS_DIR_PATHLIB, \
    WORK_DIR_PATHLIB
from hpcevolution.result import Result



class Parent:
    def __init__(self, num_iter, score_strategy, game, stop_after, save_model, gvgai_version, num_threads, log_level, max_age, is_mortality,
        is_crossover, crossover_possibility, mutate_possibility, mepgd_possibility, is_mepgd, cmame):
        self.run_name = f'{game}-iter-{num_iter}-strat-{score_strategy}-stop-after-{stop_after}'
        logging.basicConfig(filename='../logs/{}.log'.format(self.run_name), level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info('Initializing parent')
        self.scoring_strategy = score_strategy
        self.score_strategy = score_strategy
        self.game = game
        self.stop_after = stop_after
        # logging.basicConfig(filename=self.run_name + '.log', level=logging.INFO if log_level == 'INFO' else logging.DEBUG)

        policy_net, init_model = get_initial_model(gvgai_version, game)
        init_iter = 1

        self.evaluated_so_far = 0
        self.total_to_evaluate = num_iter
        self.map_elites = HPCMapElites(policy_net,
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
                  gvgai_version=gvgai_version,
                  is_cmame=cmame)

    # import pickle
    # result = Result(run_data.model, '0-0-0-0-0', 10)
    # pickle.dump(result, open('/Users/bradwindsor/ms_projects/qd-gen/gameQD/hpcevolution/results/1234.result', 'wb'))
    def run(self):
        while self.evaluated_so_far < self.total_to_evaluate:
            children = self.get_available_children()
            logging.info('{} available children, {} evals run'.format(len(children), self.evaluated_so_far))
            for child_name in children:
                logging.info('Generating work for child')
                run_data = self.generate_run_data()
                self.write_work_for_child(run_data, child_name)

            logging.info('collecting results')
            results = self.collect_written_results()
            logging.info('%d results found', len(results))
            for result, file in results:
                self.update_map_elites_results(result)
                file.unlink()
                self.evaluated_so_far += 1
            logging.info('sleeping %d', SLEEP_TIME)
            time.sleep(SLEEP_TIME)

    def get_available_children(self):
        active = AVAILABLE_AGENTS_DIR_PATHLIB.glob('*' + AVAILABLE_EXTENSION)
        active_agent_ids = [agent.stem for agent in active]
        return active_agent_ids

    def collect_written_results(self):
        results_files = RESULTS_DIR_PATHLIB.glob('*.pkl')
        results = []
        for file in results_files:
            logging.info('loaded result %s', file.stem)
            results.append((pickle.load(file.open('rb')), file))
        return results

    def update_map_elites_results(self, result: Result):
        self.map_elites.update_result(result.network, result.feature, result.fitness)

    def generate_run_data(self):
        model = self.map_elites.next_model()
        return Work(model, self.score_strategy, self.game, self.stop_after, self.run_name)

    def write_work_for_child(self, work, child_name):
        path = WORK_DIR_PATHLIB / str('task-id-{}-child-{}'.format(self.evaluated_so_far, child_name))
        path = path.with_suffix('.pkl')
        # path.unlink(missing_ok=True)
        pickle.dump(work, path.open('wb'))
