import random
import torch
from collections import OrderedDict

import random
import torch
from collections import OrderedDict

class MapElite():
    
    def __init__(self, model, fitness, feature_descriptor):
        self.solutions = {}
        self.performances = {}
        self.model = model
        self.fitness = fitness
        self.feature_descriptor = feature_descriptor
        
    def random_variation(self, is_crossover, mut_prob, cross_prob):
        if is_crossover and len(self.solutions) > 1:
            inds = random.sample(list(self.solutions.items()), 2)
            ind = crossover(inds[0][1], inds[1][1], cross_prob)
        else:
            ind = random.choice(list(self.solutions.items()))[1]
        return mutation(ind, mut_prob)
    
    def run(self, env, num_initial, iteration, is_crossover, mut_prob, cross_prob):
        for i in range(iteration):
            if i < num_initial:
                self.model.__init__()
                x = self.model.state_dict()
            else:
                x = self.random_variation(is_crossover, mut_prob, cross_prob)
            self.model.load_state_dict(x, strict = True)
            b = self.feature_descriptor(x)
            p = self.fitness(self.model, env)
            if b not in self.performances or self.performances[b] < p:
                self.performances[b] = p
                self.solutions[b] = x

def mutation(state, mut_prob):
    states = list(state.items())
    new_state = OrderedDict()
    for l, s in states:
        new_state[l] = torch.where(torch.rand_like(s) > mut_prob, torch.randn_like(s), s)
    return new_state

def crossover(state1, state2, cross_prob):
    states1 = list(state1.items())
    new_state = OrderedDict()
    for l, s in states1:
        new_state[l] = torch.where(torch.rand_like(s) > cross_prob, s, state2[l])
    return new_state