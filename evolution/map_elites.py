import random
import torch

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
    
    def random_variation(self, is_crossover):
        if is_crossover and len(self.solutions)>2:
            ind = random.sample(list(self.solutions.items()), 2)
            ind = self.crossover(ind[0][1], ind[1][1])
        else:
            ind = random.choice(list(self.solutions.items()))[1]
        return self.mutation(ind)
    
    def mutation(self, state):
        states = list(state.items())
        new_state = {}
        for l, x in states:
            if l[0:4] == "conv":
                new_state[l] = torch.where(torch.rand_like(x) > self.mutate_poss, torch.randn_like(x), x)
            else:
                new_state[l] = x
        return new_state
    
    def crossover(self, x1, x2):
        states1 = list(x1.items())
        states2 = list(x2.items())
        child = {}
        for s1, s2 in zip(states1, states2):
            l_1, s_1 = s1
            l_2, s_2 = s2
            if l_1[0:4] == "conv:":
                child[l_1] = torch.where(torch.rand_like(s_1) > self.cross_poss, s_1, s_2)
            else:
                child[l_1] = random.choice(s_1, s_2)
        return child
    
    def run(self, game_level, is_crossover):
        for i in range(self.num_iter):
            if i < self.num_initial_solutions:
                self.model.__init__(self.init_model)
                x = self.model.state_dict()
            else:
                x = self.random_variation(is_crossover)
            self.model.load_state_dict(x)
            if self.fitness_feature is not None:
                performance, features = self.fitness_feature(self.model)
            else:
                features = self.feature_descriptor(x)
                performance = self.fitness(self.model, game_level)
            if features not in self.performances or self.performances[features] < performance:
                self.performances[features] = performance
                self.solutions[features] = x
        return self.performances, self.solutions
