import time

import torch
import pickle
from pathlib import Path

from evolution.hpc_map_elites import HPCMapElites
from hpcevolution.constants import SLEEP_TIME, ACTIVE_AGENTS_DIR_PATHLIB, ACTIVE_EXTENSION, RESULTS_DIR_PATHLIB, \
    WORK_DIR_PATHLIB
from hpcevolution.result import Result


class Parent:
    def __init__(self):
        self.map_elites = HPCMapElites()

    def run(self):
        while True:
            children = self.get_available_children()
            for child_name in children:
                run_data = self.generate_run_data()
                self.write_work_for_child(run_data, child_name)

            results = self.collect_written_results()
            for result in results:
                self.update_map_elites_results(result)

            time.sleep(SLEEP_TIME)

    def get_available_children(self):
        active = ACTIVE_AGENTS_DIR_PATHLIB.glob('*' + ACTIVE_EXTENSION)
        active_agent_ids = [agent.stem for agent in active]
        return active_agent_ids

    def collect_written_results(self):
        results_files = RESULTS_DIR_PATHLIB.glob('*.pkl')
        results = []
        for file in results_files:
            results.append(pickle.load(file.open()))
        return results

    def update_map_elites_results(self, result: Result):
        self.map_elites.update_result(result.network, result.feature, result.feature)

    def generate_run_data(self):
        model = self.map_elites.next_model()
        return model


    def write_work_for_child(self, run_data, child_name):
        path = WORK_DIR_PATHLIB / child_name  + '.model'
        torch.save(run_data, str(path))

