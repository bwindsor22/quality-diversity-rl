import time

from hpcevolution.constants import SLEEP_TIME


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
        pass

    def write_work_for_child(self):
        # WRITE NEURAL NETWORK
        pass

    def update_map_elites_results(self, result):
        pass

    def collect_written_results(self):
        pass
