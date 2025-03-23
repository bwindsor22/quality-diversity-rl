import logging
import random

import torch

from evolution.cmame.cmame import CMAEmitters


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
                 gvgai_version=None,
                 is_cmame=False):

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
        self.feature_descriptor = feature_descriptor
        self.fitness_feature = fitness_feature
        self.gvgai_version = gvgai_version
        self.normal_dist_variance = 0.03
        self.log_counts = 1000  # number of times to log intermediate results

        self.cmame = is_cmame
        if self.cmame:
            self.model.__init__(*self.init_model)
            initial_state_dict = self.model.state_dict()
            self.emitters = CMAEmitters(initial_state_dict)

        self.cmame = is_cmame
        if self.cmame:
            self.model.__init__(*self.init_model)
            initial_state_dict = self.model.state_dict()
            self.emitters = CMAEmitters(initial_state_dict)

    def random_variation(self):
        logging.debug('doing random varation')
        if self.is_crossover and len(self.solutions) >= 2:
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
            # new_state[l] = torch.where(torch.rand_like(x) > self.mutate_poss, torch.randn_like(x), x)
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

    def check_mortality(self):
        for key, value in list(self.ages.items()):
            if value >= self.max_age:
                # print("Agent Died")
                del self.performances[key]
                del self.solutions[key]
                del self.ages[key]
            else:
                self.ages[key] += 1

