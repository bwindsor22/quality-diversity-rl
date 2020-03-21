class DME(object):
    
    def __init__(self, 
                 model, 
                 init_model,
                 init_iter, 
                 num_iter,
                 F,
                 cross_poss,
                 evaluate):
        
        self.solutions = {}
        self.performances = {}
        self.model = model
        self.init_model = init_model
        self.num_initial_solutions = init_iter
        self.num_iter = num_iter
        self.F = F
        self.cross_poss = cross_poss
        self.evaluate = evaluate
    
    def selection(self):
        inds = random.sample(list(self.solutions.items()), 4)
        r1 = inds[0][1]
        r2 = inds[1][1]
        r3 = inds[2][1]
        x = inds[3][1]
        return x, r1, r2, r3
    
    def mutation(self, x, r1, r2, r3):
        x = list(x.items())
        r1 = list(r1.items())
        r2 = list(r2.items())
        r3 = list(r3.items())
        v = {}
        for x1, s1, s2, s3 in zip(x, r1, r2, r3):
            l,x_1 = x1
            _,s_1 = s1
            _,s_2 = s2
            _,s_3 = s3
            if l[0:4] == "conv":
                print("Mutating")
                v[l] = s_1 + self.F*torch.sub(s_2,s_3)
            else:
                v[l] = x_1
        return v
    
    def crossover(self, x, v):
        x = list(x.items())
        v = list(v.items())
        u = {}
        layer_index = range(0, len(x))
        selected = random.sample(layer_index, 1)
        layer_iter = 0
        for x1, v1 in zip(x, v):
            l, x_1 = x1
            _, v_1 = v1
            if np.random.uniform() < self.cross_poss or  layer_iter == selected:
                print("Crossovering")
                u[l] = v_1
            else:
                u[l] = x_1
            layer_iter = layer_iter + 1
        return u

    def add_to_solutions(self, x):
        self.model.load_state_dict(x)
        p, b = self.evaluate(self.model)
        if b not in self.performances or self.performances[b] < p:
            self.solutions[b] = x
            self.performances[b] = p
    
    def run(self):
        
        while len(self.solutions) < self.num_initial_solutions:
            self.model.__init__(self.init_model)
            x = self.model.state_dict()
            self.add_to_solutions(x)
        print("# of solutions:",len(self.solutions))
        for i in range(self.num_iter):
            x, r1, r2, r3 = self.selection()
            v = self.mutation(x, r1, r2, r3)
            u = self.crossover(x, v)
            self.add_to_solutions(u)
    
        return self.performances, self.solutions