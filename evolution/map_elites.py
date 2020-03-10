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
                 mutate_poss,
                 cross_poss,
                 fitness=None,
                 feature_descriptor=None,
                 fitness_feature=None,
                 gvgai_version=None):

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
        self.gvgai_version = gvgai_version
        self.log_counts = 100 # number of times to log intermediate results

    def random_variation(self, is_crossover):
        logging.debug('doing random varation')
        if is_crossover and len(self.solutions)>2:
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

    def me_iteration(self, is_crossover, env_maker, counter):
        logging.debug('acquiring maker')
        env_maker.acquire()
        logging.debug('starting new ME iteration')
        if len(self.solutions) < self.num_initial_solutions:
            self.model.__init__(*self.init_model)
            x = self.model.state_dict()
        else:
            x = self.random_variation(is_crossover)
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

        counter.increment()
        logging.debug('releasing maker')
        env_maker.release()
    
    def run(self, thread_pool_size, is_crossover=False):
        evaluations_run = 0
        main_thread = threading.currentThread()

        env_makers = [LockableResource(CachingEnvironmentMaker(version=self.gvgai_version))
                      for _ in range(thread_pool_size)]

        C =  Counter()
        sleep_time = 7
        begin_ramping = 20
        times_to_log = [int(i) for i in np.linspace(0, self.num_iter, self.log_counts).tolist()]
        while evaluations_run < self.num_iter:
            evaluations_run = C.get_value()

            #startup ramp
            if abs(evaluations_run - begin_ramping) < 5:
                logging.info('reducing sleep time')
                sleep_time = 4
            elif abs(evaluations_run - begin_ramping - 10) < 5:
                logging.info('reducing sleep time again')
                sleep_time = 3
            elif abs(evaluations_run - begin_ramping - 20) < 5:
                logging.info('reducing sleep time again again')
                sleep_time = 1.5


            logging.debug('sleeping %d', sleep_time)
            time.sleep(sleep_time)

            # log results partway through run
            if self.num_iter > self.log_counts * 5 and len(times_to_log) > 0 and evaluations_run > times_to_log[0]:
                times_to_log.pop(0)
                logging.info('LOGGING INTERMEDIATE RESULTS {}, {}'.format(evaluations_run, str(self.performances)))
                logger = logging.getLogger()
                logger.handlers[0].flush()
                logger.handlers[1].flush()


            active_threads = [t for t in threading.enumerate() if not t is main_thread]
            num_active = len(active_threads)
            if num_active < thread_pool_size * 7:
                if evaluations_run < 50 or evaluations_run % 100 == 0:
                    logging.info('%d threads active, %d threadpool size. Starting new thread.', num_active, thread_pool_size)
                    logging.info('Map elites iterations finished: {}'.format(evaluations_run))

                unlocked_makers = [l for l in env_makers if not l.is_locked()]
                if evaluations_run < 10 or evaluations_run % 5 == 0:
                    logging.info('%d unlocked makers', len(unlocked_makers))
                if len(unlocked_makers):
                    env_maker = unlocked_makers[0]
                    t = threading.Thread(name = 'run-{}'.format(evaluations_run),
                                         target = self.me_iteration,
                                         args = [is_crossover, env_maker, C])
                    t.start()
                    logging.debug('started new thread')
                else:
                    logging.debug('No unlocked makers')
            else:
                if evaluations_run < 10 or evaluations_run % 5 == 0:
                    logging.info('%d active threads, %d thread pool size', num_active, thread_pool_size)
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
