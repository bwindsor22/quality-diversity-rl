import random
import logging
from pprint import pformat

class FeatureMap:
    def __init__(self):
        self.elite_map = {} # fitness -> individual
        self.delta_offset = 10000 # ensures new individuals go to front of queue

    def log(self):
        logging.info(pformat(self.elite_map))

    def count_elites(self):
        return len(self.elite_map.keys())

    def get_random_elite(self):
        return random.choice(list(self.elite_map.values()))

    def add_to_map(self, individual):
        added = False
        if individual.feature not in self.elite_map:
            added = True
            individual.delta = individual.fitness + self.delta_offset
            self.elite_map[individual.feature] = individual
        elif self.elite_map[individual.feature].fitness < individual.fitness:
            added = True
            individual.delta = individual.fitness - self.elite_map[individual.feature].fitness
            self.elite_map[individual.feature] = individual
        return added

    def add_individuals_to_map(self, individuals):
        for indiv in individuals:
            self.add_to_map(indiv)
        logging.info('added many to map, current keys: %s', str(self.elite_map.keys()))
