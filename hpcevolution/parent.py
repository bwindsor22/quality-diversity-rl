import time

import pickle
from pathlib import Path
from hpcevolution.constants import SLEEP_TIME, ACTIVE_AGENTS_DIR_PATHLIB, ACTIVE_EXTENSION, RESULTS_DIR_PATHLIB
from hpcevolution.result import Result


class Parent:
    def __init__(self):
        self.map_elite_results = {}

    def create_children(self):
        pass

    def run(self):
        while True:
            children = self.get_available_childrent()
            for child in children:
                self.write_work_for_child()

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

    def update_map_elites_results(self, result: List[Result]):
        pass


    def write_work_for_child(self):
        # WRITE NEURAL NETWORK
        pass


