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
                 is_mepgd,
                 mepgd_possibility,
                 fitness=None,
                 feature_descriptor=None,
                 fitness_feature=None,
                 gvgai_version=None):

        self.solutions = {}
        self.performances = {}
        self.secondary_solutions = {}
        self.secondary_performances = {}
        self.is_mepgd = is_mepgd
        self.mepgd_poss = mepgd_possibility
        self.ages = {}
        self.max_age = max_age
        self.model = model
        self.init_model = init_model
        self.num_initial_solutions = init_iter
        self.num_iter = num_iter
        self.is_crossover = is_crossover
        self.is_mortality = is_mortality
        self.cross_poss = cross_poss
        self.mutate_poss = mutate_poss
        self.fitness = fitness
        self.feature_descriptor = feature_descriptor
        self.fitness_feature = fitness_feature
        self.gvgai_version = gvgai_version
        self.log_counts = 1000 # number of times to log intermediate results

    def random_variation(self):
        logging.debug('doing random varation')
        if self.is_crossover and len(self.solutions)>=2:
            if self.is_mepgd == False:
                ind = random.sample(list(self.solutions.items()), 2)
                ind = self.crossover(ind[0][1], ind[1][1])
            elif len(self.secondary_solutions) > 0:
                ind = []
                ind.append(random.choice([random.choice(list(self.solutions.items())),
                                          random.choice(list(self.secondary_solutions.items()))]))
                
                ind.append(random.choice([random.choice(list(self.solutions.items())),
                                          random.choice(list(self.secondary_solutions.items()))]))
                

        elif len(self.secondary_solutions) > 0 and self.is_mepgd == True:
            ind = random.choice([random.choice(list(self.solutions.values())),
                                random.choice(list(self.secondary_solutions.values()))])
        else:
            ind = random.choice(list(self.solutions.values()))
        return self.mutation(ind)
    
    def mutation(self, state):
        logging.debug('doing mutation')
        states = list(state.items())
        new_state = {}
        for l, x in states:
            if l[-6:] == "weight" or l[-4:] == "bias":
                new_state[l] = torch.where(torch.rand_like(x) > self.mutate_poss, torch.randn_like(x), x)
            else:
                new_state[l] = x
        return new_state
    
    def crossover(self, x1, x2):
        logging.debug('doing crossover')
        if random.random() > self.cross_poss:
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
        #print("Checking age")
        for key,value in list(self.ages.items()):
            if value >= self.max_age:
                #print("Agent Died")
                del self.performances[key]
                del self.solutions[key]
                del self.ages[key]
            else:
                self.ages[key] += 1
    
    def me_iteration(self,env_maker, counter):
        env_maker.acquire()
        if self.is_mortality == True:
            #print("morta",self.is_mortality)
            self.check_mortality()
        if len(self.solutions) < self.num_initial_solutions:
            self.model.__init__(*self.init_model)
            x = self.model.state_dict()
            #print("CREATED")
        else:
            #print("VARIATING")

            x = self.random_variation()
            
        self.model.load_state_dict(x)
        if self.fitness_feature is not None:
            performance, feature = self.fitness_feature(self.model, env_maker) if env_maker  else\
                                    self.fitness_feature(self.model)
        else:
            feature = self.feature_descriptor(x)
            performance = self.fitness(self.model)
        if feature not in self.performances or self.performances[feature] < performance:
            logging.debug('Found better performance for feature: {}, new score: {}'.format(feature, performance))
            self.performances[feature] = performance
            self.solutions[feature] = x
        elif self.is_mepgd == True:
            if random.random() > self.mepgd_poss:
                logging.debug('Saving secondary performance for feature: {}, new score: {}'.format(feature, performance))
                self.secondary_performances[feature] = performance
                self.secondary_solutions[feature] = x

        logging.debug('releasing maker')
        self.ages[feature] = 0
    
        counter.increment()
        env_maker.release()
    
    def run(self, thread_pool_size):
        evaluations_run = 0
        main_thread = threading.currentThread()

        env_makers = [LockableResource(CachingEnvironmentMaker(version=self.gvgai_version))
                      for _ in range(thread_pool_size)]

        C =  Counter()
        sleep_time = 7
        end_sleep_time = 1.5
        ramp_steps = 10
        ramp_by = (sleep_time - end_sleep_time) / ramp_steps
        begin_ramping = 7
        end_ramping = 60
        
        #start at a slower speed, ramp up slowly. Calculate times to ramp.
        mid_ramp = begin_ramping + int((begin_ramping + end_ramping) / 2)
        times_to_ramp = [int(i) for i in np.linspace(begin_ramping, mid_ramp, int(ramp_steps * 0.75) ).tolist()]
        ttr_2 = [int(i) for i in np.linspace(mid_ramp, end_ramping, int(ramp_steps * 0.25) + 1 ).tolist()]
        times_to_ramp.extend(ttr_2[1:])

        logging.info('will speed up sleep time at eval runs %s', str(times_to_ramp))
        times_to_log = [int(i) for i in np.linspace(0, self.num_iter, self.log_counts).tolist()]
        i = 0
        unlocked_makers = []
        while evaluations_run < self.num_iter:
            evaluations_run = C.get_value()

            if evaluations_run <= end_ramping + 50:
                logging.info('starting new iter')
                while times_to_ramp and evaluations_run >= times_to_ramp[0]:
                    times_to_ramp.pop(0)
                    sleep_time = max(end_sleep_time, sleep_time - ramp_by)
                    logging.info('reducing sleep')
                logging.info('sleeping %d', sleep_time)

            time.sleep(sleep_time)

            active_threads = [t for t in threading.enumerate() if not t is main_thread]
            num_active = len(active_threads)
            if True: # num_active < thread_pool_size * 8:
                if evaluations_run < 100:
                    logging.info('%d threads active, %d threadpool size.', num_active, thread_pool_size)
                    logging.info('Map elites iterations finished: {}'.format(evaluations_run))

                unlocked_makers = [l for l in env_makers if not l.is_locked()]
                if evaluations_run < 100 or (evaluations_run % 10 == 0 and evaluations_run < 200):
                    logging.info('%d unlocked reusable level makers', len(unlocked_makers))
                if len(unlocked_makers):
                    env_maker = unlocked_makers[0]
                    t = threading.Thread(name = 'run-{}'.format(evaluations_run),
                                         target = self.me_iteration,
                                         args = [env_maker, C])
                    t.start()
                    logging.debug('started new thread')
                else:
                    logging.debug('No unlocked makers')
            else:
                if evaluations_run < 100:
                    logging.info('%d active threads, %d thread pool size', num_active, thread_pool_size)
                    logging.info('Map elites iterations finished: {}'.format(evaluations_run))

            # log results partway through run
            if len(times_to_log) > 0 and evaluations_run > times_to_log[0]:
                times_to_log.pop(0)
                logging.info('LOGGING INTERMEDIATE RESULTS log-time iters: {}, evals: {}, active: {}, unlocked: {}, performs: {}'
                             .format(i, evaluations_run, num_active, len(unlocked_makers), str(self.performances)))
            if i % 1000 == 0:
                logging.info('LOGGING INTERMEDIATE RESULTS iter-time iters: {}, evals: {}, active: {}, unlocked: {}, performs: {}'
                             .format(i, evaluations_run, num_active, len(unlocked_makers), str(self.performances)))

            i += 1

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
