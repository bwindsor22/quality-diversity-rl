import random
import time
import torch
import logging
import numpy as np
import threading

from environment_utils.utils import get_run_file_name
from models.caching_environment_maker import CachingEnvironmentMaker
from models.lockable_resource import LockableResource


class MapElites(object):
    
    def __init__(self, 
                 model, 
                 init_model,
                 init_iter, 
                 num_iter,
                 is_crossover,
                 mutate_poss,
                 cross_poss,
                 is_mortality,
                 max_age,
                 fitness=None,
                 feature_descriptor=None,
                 fitness_feature=None,
                 gvgai_version=None):

        self.solutions = {}
        self.performances = {}
        self.ages = {}
        self.max_age = max_age
        self.model = model
        self.init_model = init_model
        self.num_initial_solutions = init_iter
        self.num_iter = num_iter
        self.is_crossover = is_crossover
        self.mutate_poss = mutate_poss
        self.cross_poss = cross_poss
        self.fitness = fitness
        self.feature_descriptor = feature_descriptor
        self.fitness_feature = fitness_feature
        self.gvgai_version = gvgai_version
        self.log_counts = 10 # number of times to log intermediate results

    def random_variation(self):
        logging.debug('doing random varation')
        if self.is_crossover and len(self.solutions)>2:
            ind = random.sample(list(self.solutions.items()), 2)
            ind = self.crossover(ind[0][1], ind[1][1])
        else:
            ind = random.choice(list(self.solutions.values()))
        return self.mutation(ind)
    
    def mutation(self, state):
        logging.debug('doing mutation')
        states = list(state.items())
        new_state = {}
        for l, x in states:
            if l[0:4] == "conv":
                new_state[l] = torch.where(torch.rand_like(x) > self.mutate_poss, torch.randn_like(x), x)
            else:
                new_state[l] = x
        return new_state
    
    def crossover(self, x1, x2):
        logging.debug('doing crossover')
        if random.random() < self.cross_poss:
            return random.choice([x1, x2])
        states1 = list(x1.items())
        states2 = list(x2.items())
        child = {}
        for s1, s2 in zip(states1, states2):
            l_1, s_1 = s1
            l_2, s_2 = s2
            logging.debug("Crossovering")
            child[l_1] = random.choice([s_1, s_2])
        return child

    
    def check_mortality(self):
        for key,value in list(self.ages.items()):
            if value >= self.max_age:
                del self.performances[key]
                del self.solutions[key]
                del self.ages[key]
            else:
                self.ages[key] += 1
    
    def me_iteration(self, env_maker, counter):
        env_maker.acquire()

        if len(self.solutions) < self.num_initial_solutions:
            self.model.__init__(*self.init_model)
            x = self.model.state_dict()
        else:
            if self.is_mortality == True:
                self.check_mortality()
            x = self.random_variation(self.is_crossover)
        self.model.load_state_dict(x)
        if self.fitness_feature is not None:
            performance, feature = self.fitness_feature(self.model, env_maker) if env_maker  else\
                                    self.fitness_feature(self.model)
        else:
            feature = self.feature_descriptor(x)
            performance = self.fitness(self.model)
        if feature not in self.performances or self.performances[feature] < performance:
            logging.info('Found better performance for feature: {}, new score: {}'.format(feature, performance))
            self.performances[feature] = performance
            self.solutions[feature] = x 
            self.ages[feature] = 0

        counter.increment()
        env_maker.release()
    
    def run(self, thread_pool_size):
        evaluations_run = 0
        main_thread = threading.currentThread()

        env_makers = [LockableResource(CachingEnvironmentMaker(version=self.gvgai_version))
                      for _ in range(thread_pool_size)]

        C =  Counter()
        while evaluations_run < self.num_iter:
            evaluations_run = C.get_value()
            time.sleep(7)

            # log results partway through run
            if self.num_iter > self.log_counts * 5 and evaluations_run % int(self.num_iter / self.log_counts) == 0 and self.num_iter is not 0:
                logging.info('LOGGING INTERMEDIATE RESULTS {}'.format(str(self.performances)))


            active_threads = [t for t in threading.enumerate() if not t is main_thread]
            num_active = len(active_threads)
            if num_active < thread_pool_size:
                if evaluations_run < 100 or evaluations_run % 100 == 0:
                    logging.info('%d threads active, %d threadpool size. Starting new thread.', num_active, thread_pool_size)
                    logging.info('Map elites iterations finished: {}'.format(evaluations_run))

                unlocked_makers = [l for l in env_makers if not l.is_locked()]
                if len(unlocked_makers):
                    env_maker = unlocked_makers[0]
                    t = threading.Thread(name = 'run-{}'.format(evaluations_run),
                                         target = self.me_iteration,
                                         args = [self.is_crossover, env_maker, C])
                    t.start()
            else:
                logging.info('Map elites iterations finished: {}'.format(evaluations_run))

        return self.performances, self.solutions


class Counter(object):
    def __init__(self, start=0):
        self.lock = threading.Lock()
        self.value = start

    def increment(self):
        logging.debug('Waiting for lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired lock')
            self.value = self.value + 1
        finally:
            self.lock.release()

    def get_value(self):
        return self.value
