  
import logging

from evolution.dme import DME


class HPC_DME(DME):

    def next_model(self):
        if len(self.solutions) < self.num_initial_solutions:
            self.model.__init__(*self.init_model)
            model_state = self.model.state_dict()
        else:
            x, r1, r2, r3 = self.selection()
            v = self.mutation(r1, r2, r3)
            model_state = self.crossover(x, v)
        return model_state


    def update_result(self, x, b, p):
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
