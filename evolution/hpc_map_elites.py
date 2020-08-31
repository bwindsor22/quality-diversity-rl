import logging

from evolution.map_elites import MapElites
import torch

class HPCMapElites(MapElites):

    def next_model(self):
        if self.cmaes:
            model_state = self.cmaes.ask()
        elif self.preloaded_model_path and len(self.solutions) < self.num_initial_solutions:
            print('loading preloaded')
            state_dict = torch.load(self.preloaded_model_path)
            # self.model.load_state_dict(state_dict)
            return state_dict
        elif len(self.solutions) < self.num_initial_solutions:
            self.model.__init__(*self.init_model)
            model_state = self.model.state_dict()
        else:
            model_state = self.random_variation()
        return model_state


    def update_result(self, network, feature, fitness):
        logging.info('Updating feature {}, performance {}'.format(feature, fitness))

        if self.cmaes:
            self.cmaes.tell(network, feature, fitness)
        elif feature not in self.performances or self.performances[feature] < fitness:
            logging.info('Found better performance for feature: {}, new score: {}'.format(feature, fitness))
            self.performances[feature] = fitness
            self.solutions[feature] = network
        logging.info('updated map elites with result')
