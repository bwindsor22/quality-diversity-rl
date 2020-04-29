import random
import torch
import logging
from models.caching_environment_maker import CachingEnvironmentMaker

class DME(object):
    
    def __init__(self, 
                 model, 
                 init_model,
                 init_iter, 
                 num_iter,
                 F,
                 cross_poss,
                 evaluate,
                 gvgai_version):
        
        self.solutions = {}
        self.performances = {}
        self.model = model
        self.init_model = init_model
        self.num_initial_solutions = init_iter
        self.num_iter = num_iter
        self.F = F
        self.cross_poss = cross_poss
        self.evaluate = evaluate
        self.gvgai_version = gvgai_version
    
    def selection(self):
        inds = []
        if len(self.solutions) > 4:
            lists = random.sample(list(self.solutions.items()), 4)
            for l in lists:
                inds.append(random.choice(l[1]))
        else:
            counts = [0 for i in range(len(self.solutions))]
            i = 0
            j = 0
            while i < 4:
                if len(list(self.solutions.items())[j][1]) > counts[j]:
                    counts[j] = counts[j] + 1
                    i = i + 1
                j = (j + 1) % len(self.solutions)
            for c, s in zip(counts, list(self.solutions.items())):
                inds.extend(random.sample(s[1], c))
        r1 = inds[0]
        r2 = inds[1]
        r3 = inds[2]
        x = inds[3]
        return x, r1, r2, r3
    
    def mutation(self, r1, r2, r3):
        r1 = list(r1.items())
        r2 = list(r2.items())
        r3 = list(r3.items())
        v = {}
        logging.debug("Mutating")
        for s1, s2, s3 in zip(r1, r2, r3):
            l,s_1 = s1
            _,s_2 = s2
            _,s_3 = s3
            v[l] = s_1 + self.F*torch.sub(s_2,s_3)
        return v
    
    def crossover(self, x, v):
        x = list(x.items())
        v = list(v.items())
        u = {}
        logging.debug("Crossovering")
        for x1, v1 in zip(x, v):
            l, param1 = x1
            _, param2 = v1
            u[l] = torch.where(torch.rand_like(param1) < self.cross_poss, param2, param1)
        return u

    def add_to_solutions(self, x, env_maker):
        p, b = self.evaluate(policy_net = self.model, env_maker=env_maker)
        if b not in self.performances:
            self.solutions[b] = [x]
            self.performances[b] = [p]
        elif len(self.performances[b]) < 10:
            self.solutions[b].append(x)
            self.performances[b].append(p)
        elif self.performances[b][0] < p:
            self.solutions[b][0] = x
            self.performances[b][0] = p
        index = range(len(self.solutions[b]))
        self.performances[b] = [x for x,_ in sorted(zip(self.performances[b], index))]
        sorted_index = [y for _,y in sorted(zip(self.performances[b], index))]
        self.result = self.solutions[b]
        for i, j in zip(index, sorted_index):
            self.result[i] = self.solutions[b][j]
        self.solutions[b] = self.result
        
    
    def run(self):
        Envmaker = CachingEnvironmentMaker(version=self.gvgai_version)
        for i in range(self.num_initial_solutions):
            self.model.__init__(*self.init_model)
            x = self.model.state_dict()
            self.add_to_solutions(x, Envmaker)
        for i in range(self.num_iter):
            if i % 100 == 0:
                logging.info('Iteration: %d', i)
                logging.info(str(self.performances))
            x, r1, r2, r3 = self.selection()
            v = self.mutation(r1, r2, r3)
            u = self.crossover(x, v)
            self.add_to_solutions(u, Envmaker)
        return self.performances, self.solutions
