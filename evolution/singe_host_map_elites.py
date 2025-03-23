import logging
import random
import threading
import time

import numpy as np
#import psutil

from evolution.map_elites import MapElites
from models.caching_environment_maker import CachingEnvironmentMaker
from models.lockable_resource import LockableResource


class SingleHostMapElites(MapElites):

    def me_iteration(self, env_maker, counter):
        env_maker.acquire()
        if self.is_mortality == True:
            # print("morta",self.is_mortality)
            self.check_mortality()
        if len(self.solutions) < self.num_initial_solutions:
            self.model.__init__(*self.init_model)
            model_state = self.model.state_dict()
            # print("CREATED")
        elif self.cmame:
            model_state = self.emitters.ask()
        else:
            # print("VARIATING")

            model_state = self.random_variation()

        self.model.load_state_dict(model_state)
        if self.fitness_feature is not None:
            performance, feature = self.fitness_feature(self.model, env_maker) if env_maker else \
                self.fitness_feature(self.model)
        else:  # not used
            feature = self.feature_descriptor(model_state)
            performance = self.fitness(self.model)

        if self.cmame:
            self.emitters.tell(feature, model_state, performance)

        if feature not in self.performances or self.performances[feature] < performance:
            logging.debug('Found better performance for feature: {}, new score: {}'.format(feature, performance))
            self.performances[feature] = performance
            self.solutions[feature] = model_state
        elif self.is_mepgd == True:
            if random.random() > self.mepgd_poss:
                logging.debug(
                    'Saving secondary performance for feature: {}, new score: {}'.format(feature, performance))
                self.secondary_performances[feature] = performance
                self.secondary_solutions[feature] = model_state

        logging.debug('releasing maker')
        self.ages[feature] = 0

        counter.increment()
        env_maker.release()

    def run(self, thread_pool_size):
        evaluations_run = 0
        main_thread = threading.currentThread()
        # main_thread = multiprocessing.current_process()

        env_makers = [LockableResource(CachingEnvironmentMaker(version=self.gvgai_version))
                      for _ in range(thread_pool_size)]

        C = Counter()
        sleep_time = 3
        end_sleep_time = 1.5
        ramp_steps = 10
        ramp_by = (sleep_time - end_sleep_time) / ramp_steps
        begin_ramping = 7
        end_ramping = 60

        # start at a slower speed, ramp up slowly. Calculate times to ramp.
        mid_ramp = begin_ramping + int((begin_ramping + end_ramping) / 2)
        times_to_ramp = [int(i) for i in np.linspace(begin_ramping, mid_ramp, int(ramp_steps * 0.75)).tolist()]
        ttr_2 = [int(i) for i in np.linspace(mid_ramp, end_ramping, int(ramp_steps * 0.25) + 1).tolist()]
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
            # active_threads = [p for p in multiprocessing.active_children() if not p is main_thread]
            num_active = len(active_threads)

            if num_active < thread_pool_size:
                if evaluations_run < 100:
                    logging.info('%d threads active', num_active)
                    logging.info('Map elites iterations finished: {}'.format(evaluations_run))

                unlocked_makers = [l for l in env_makers if not l.is_locked()]
                if evaluations_run < 100 or (evaluations_run % 10 == 0 and evaluations_run < 200):
                    logging.info('%d unlocked reusable level makers, %d total', len(unlocked_makers), thread_pool_size)
                if len(unlocked_makers):
                    env_maker = unlocked_makers[0]
                    t = threading.Thread(name='run-{}'.format(evaluations_run),
                                         target=self.me_iteration,
                                         args=[env_maker, C])
                    # t = multiprocessing.Process(
                    #     name='run-{}'.format(evaluations_run),
                    #     target=self.me_iteration,
                    #     args=[env_maker, C]
                    # )
                    t.start()
                    logging.debug('started new thread')
                else:
                    logging.debug('No unlocked makers')

            if i != 0 and i % 20 == 0:
                logging.info('CYCLING ENVIRONMENTS')
                #logging.info('virtual mem %s', str(psutil.virtual_memory()))
                #logging.info('swap mem %s', str(psutil.swap_memory()))
                active_threads = [t for t in threading.enumerate() if not t is main_thread]
                # active_threads = [p for p in multiprocessing.active_children() if not p is main_thread]
                num_active = len(active_threads)
                logging.info('%d threads active', num_active)
                for i, th in enumerate(active_threads):
                    logging.info('joining thread %d', i)
                    th.join()

                for i, maker in enumerate(env_makers):
                    logging.info('deleting maker %d', i)
                    maker.acquire()
                    del maker
                del env_makers
                logging.info('sleeping %d', 5)
                time.sleep(5)
                #logging.info('virtual mem %s', str(psutil.virtual_memory()))
                #logging.info('swap mem %s', str(psutil.swap_memory()))
                active_threads = [t for t in threading.enumerate() if not t is main_thread]
                # active_threads = [p for p in multiprocessing.active_children() if not p is main_thread]
                num_active = len(active_threads)
                logging.info('%d threads active', num_active)
                logging.info('recreating makers')
                env_makers = [LockableResource(CachingEnvironmentMaker(version=self.gvgai_version))
                              for _ in range(thread_pool_size)]

            # log results partway through run
            if len(times_to_log) > 0 and evaluations_run > times_to_log[0]:
                times_to_log.pop(0)
                logging.info(
                    'LOGGING INTERMEDIATE RESULTS log-time iters: {}, evals: {}, active: {}, unlocked: {}, performs: {}'
                    .format(i, evaluations_run, num_active, len(unlocked_makers), str(self.performances)))
            if i % 1000 == 0:
                logging.info(
                    'LOGGING INTERMEDIATE RESULTS iter-time iters: {}, evals: {}, active: {}, unlocked: {}, performs: {}'
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
