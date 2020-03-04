import random
import torch
import logging
import numpy as np
from environment_utils.utils import get_run_file_name
from models.caching_environment_maker import CachingEnvironmentMaker
logging.basicConfig(filename=get_run_file_name(),level=logging.INFO)

class MapElites(object):
    
    def __init__(self, 
                 model, 
                 init_model,
                 init_iter, 
                 num_iter,
                 mutate_poss,
                 cross_poss,
                 fitness=None,
                 feature_descriptor=None,
                 fitness_feature=None):
        
        self.solutions = {}
        self.performances = {}
        self.model = model
        self.init_model = init_model
        self.num_initial_solutions = init_iter
        self.num_iter = num_iter
        self.mutate_poss = mutate_poss
        self.cross_poss = cross_poss
        self.fitness = fitness
        self.feature_descriptor = feature_descriptor
        self.fitness_feature = fitness_feature
        self.log_counts = 10
        self.env_maker = CachingEnvironmentMaker()
    
    def random_variation(self, is_crossover):
        logging.info('doing random varation')
        if is_crossover and len(self.solutions)>2:
            ind = random.sample(list(self.solutions.items()), 2)
            ind = self.crossover(ind[0][1], ind[1][1])
        else:
            ind = random.choice(list(self.solutions.values()))
        return self.mutation(ind)
    
    def mutation(self, state):
        logging.info('doing mutation')
        states = list(state.items())
        new_state = {}
        for l, x in states:
            if l[0:4] == "conv":
                new_state[l] = torch.where(torch.rand_like(x) > self.mutate_poss, torch.randn_like(x), x)
            else:
                new_state[l] = x
        return new_state
    
    def crossover(self, x1, x2):
        logging.info('doing crossover')
        if random.random() < self.cross_poss:
            return random.choice([x1, x2])
        states1 = list(x1.items())
        states2 = list(x2.items())
        child = {}
        for s1, s2 in zip(states1, states2):
            l_1, s_1 = s1
            l_2, s_2 = s2
            print("Crossovering")
            child[l_1] = random.choice([s_1, s_2])
        return child
    
    def run(self, game_level=None, is_crossover=True):
        logging.info('Running map elites for iter: {}'.format(self.num_iter))
        for i in range(self.num_iter):
            logging.info('Beginning map elites iter: {}'.format(i))
            if i < self.num_initial_solutions:
                self.model.__init__(*self.init_model)
                x = self.model.state_dict()
            else:
                x = self.random_variation(is_crossover)
            self.model.load_state_dict(x)
            if self.fitness_feature is not None:
                performance, feature = self.fitness_feature(self.model, env_maker=self.env_maker)
            else:
                feature = self.feature_descriptor(x)
                performance = self.fitness(self.model, game_level)
            if feature not in self.performances or self.performances[feature] < performance:
                logging.info('Found better performance for feature: {}, new score: {}'.format(feature, performance))
                self.performances[feature] = performance
                self.solutions[feature] = x
            if self.num_iter > self.log_counts * 5 and i % int(self.num_iter / self.log_counts) == 0:
                logging.info('LOGGING INTERMEDIATE RESULTS {}'.format(str(self.performances)))
        return self.performances, self.solutions
