import logging

from evolution.map_elites import MapElites


class HPCMapElites(MapElites):

    def next_model(self):
        if len(self.solutions) < self.num_initial_solutions:
            self.model.__init__(*self.init_model)
            model_state = self.model.state_dict()
            # print("CREATED")
        elif self.cmame:
            model_state = self.emitters.ask()
        else:
            # print("VARIATING")

            model_state = self.random_variation()
        return model_state


    def update_result(self, network, feature, fitness):
        if feature not in self.performances or self.performances[feature] < fitness:
            logging.debug('Found better performance for feature: {}, new score: {}'.format(feature, fitness))
            self.performances[feature] = fitness
            self.solutions[feature] = network
        logging.info('Updated result')
