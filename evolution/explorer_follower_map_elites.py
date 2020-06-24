from pprint import pformat
import logging
import random

import torch

from evolution.optimizing_emitter.caching_pop_updater import CachingPopUpdater

EXPLORE_STATE = 'explore'
FOLLOW_STATE = 'follow'

switch = {
    EXPLORE_STATE: FOLLOW_STATE,
    FOLLOW_STATE: EXPLORE_STATE
}

class ExplorerFollowerMapElites(object):

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
                 gvgai_version=None,
                 is_cmame=False):

        self.explorer_solutions = {}
        self.explorer_performances = {}
        self.follower_solutions = {}
        self.follower_performances = {}
        self.follower_timesteps = {}

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
        self.fitness = fitness  # not used
        self.feature_descriptor = feature_descriptor
        self.fitness_feature = fitness_feature
        self.gvgai_version = gvgai_version
        self.normal_dist_variance = 0.03
        self.log_counts = 1000  # number of times to log intermediate results

        self.state = EXPLORE_STATE
        self.explore_headway = 2000
        self.explore_stop_after = 60000
        self.follow_stop_after = 20000
        self.allow_score_to_drop = 0

        self.iter_total = 0
        self.iter_in_state = 0

        self.cmame = is_cmame
        if self.cmame:
            self.model.__init__(*self.init_model)
            initial_state_dict = self.model.state_dict()
            self.emitters = CachingPopUpdater(initial_state_dict)

    def random_variation(self, solution_map):
        logging.debug('doing random varation')
        if self.is_crossover and len(solution_map) >= 2:
            if self.is_mepgd == False:
                ind = random.sample(list(solution_map.items()), 2)
                ind = self.crossover(ind[0][1], ind[1][1])
        else:
            ind = random.choice(list(solution_map.values()))
        return self.mutation(ind)

    def mutation(self, state):
        logging.debug('doing mutation')
        states = list(state.items())
        new_state = {}
        for l, x in states:
            if l[-6:] == "weight" or l[-4:] == "bias":
                new_state[l] = torch.where(torch.rand_like(x) > self.mutate_poss,
                                           torch.randn_like(x) * self.normal_dist_variance + x,
                                           x)
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


    def next_model_and_hyperparams(self):
        # if self.cmame:
        #     model_state = self.emitters.ask()
        # el
        self.set_correct_state()
        if self.state == EXPLORE_STATE:
            if len(self.explorer_solutions) < self.num_initial_solutions:
                self.model.__init__(*self.init_model)
                model_state = self.model.state_dict()
            else:
                model_state = self.random_variation(self.explorer_solutions)
        else: # follow state
            follow_count = len(self.follower_solutions.values())
            sample_from_explore = follow_count == 0
            sample_map = self.explorer_solutions if sample_from_explore else self.follower_solutions
            model_state = self.random_variation(sample_map)

        stop_after = self.explore_stop_after if self.state == EXPLORE_STATE else self.follow_stop_after

        self.iter_total += 1
        self.iter_in_state += 1
        return model_state, stop_after


    def set_correct_state(self):
        if self.iter_total <= self.explore_headway:
            self.state = EXPLORE_STATE
        elif self.iter_in_state >= 1000 and self.state == EXPLORE_STATE:
            logging.info('Switching to follow at {} iters'.format(self.iter_total))
            self.state = FOLLOW_STATE
            self.iter_in_state = 0
        elif self.iter_in_state >= 2000 and self.state == FOLLOW_STATE:
            logging.info('Switching to explore at {} iters'.format(self.iter_total))
            self.state = EXPLORE_STATE
            self.iter_in_state = 0


    def update_result(self, network, feature, fitness, eval_steps):
        logging.info('Updating feature {}, performance {}, timestep: {}'.format(feature, fitness, eval_steps))

        # if self.cmame:
        #     self.emitters.tell(feature, network, fitness)
        # el
        if feature not in self.explorer_performances or self.explorer_performances[feature] < fitness:
            logging.info('Found better explorer performance for feature: {}, new score: {}'.format(feature, fitness))
            self.explorer_performances[feature] = fitness
            self.explorer_solutions[feature] = network

        if feature not in self.follower_performances or self.follower_timesteps[feature] > eval_steps:
            log_str = f'Found better follower runtime for feature: {feature},'
            if feature in self.follower_performances:
                log_str += f' old score: {self.follower_performances[feature]},'
                f' old timestep: {self.follower_timesteps[feature]},'
            logging.info(log_str +
                f' new score: {fitness},'
                f' new timestep: {eval_steps}')
            self.follower_solutions[feature] = network
            self.follower_performances[feature] = fitness
            self.follower_timesteps[feature] = eval_steps

        logging.info('updated map elites with result')

    def log(self):
        logging.info('EXPLORER MAP')
        logging.info(pformat(self.explorer_performances))

        logging.info('FOLLOWER MAP')
        logging.info('Follower performances')
        logging.info(pformat(self.follower_performances))

        logging.info('Follower timesteps')
        logging.info(pformat(self.follower_timesteps))