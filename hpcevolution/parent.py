import time

import torch
import pickle
from pathlib import Path
import logging

from evolution.hpc_dme import HPC_DME
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

        policy_net, init_model = get_initial_model(gvgai_version, game)

        init_iter = 5
        crossover_possibility = 0.8
        F = 0.8

        self.evaluated_so_far = 0
        self.count_loops = 0
        self.total_to_evaluate = num_iter
        
        self.dme = HPC_DME(policy_net,
                  init_model,
                  init_iter,
                  num_iter,
                  F,
                  crossover_possibility,
                  evaluate = None,
                  gvgai_version=gvgai_version)

    # import pickle
    # result = Result(run_data.model, '0-0-0-0-0', 10)
    # pickle.dump(result, open('/Users/bradwindsor/ms_projects/qd-gen/gameQD/hpcevolution/results/1234.result', 'wb'))
    def run(self):
        while self.evaluated_so_far < self.total_to_evaluate:
            if self.count_loops % 50 == 0:
                logging.info('INTERMEDIATE PERFORMANCES')
                logging.info(str(self.dme.performances))
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
            self.count_loops += 1

        logging.info('Logging final results')
        logging.info(str(self.dme.performances))


    def get_available_children(self):
        active = AVAILABLE_AGENTS_DIR_PATHLIB.glob('*' + AVAILABLE_EXTENSION)
        active_agent_ids = [agent.stem for agent in active]
        return active_agent_ids

    def collect_written_results(self):
        results_files = RESULTS_DIR_PATHLIB.glob('*.pkl')
        results = []
        for file in results_files:
            logging.info('loading result %s', file.stem)
            try:
                results.append((pickle.load(file.open('rb')), file))
            except Exception as e:
                logging.info('failed to load result %s', str(e))
        logging.info('loaded')
        return results

    def update_map_elites_results(self, result: Result):
        self.dme.update_result(result.network, result.feature, result.fitness)

    def generate_run_data(self):
        model = self.dme.next_model()
        return Work(model, self.score_strategy, self.game, self.stop_after, self.run_name)

    def write_work_for_child(self, work, child_name):
        path = WORK_DIR_PATHLIB / str('task-id-{}-child-{}'.format(self.evaluated_so_far, child_name))
        path = path.with_suffix('.pkl')
        pickle.dump(work, path.open('wb'))