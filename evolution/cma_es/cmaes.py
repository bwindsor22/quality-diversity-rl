from collections import defaultdict
from math import log
from copy import deepcopy
import random
from pprint import pformat, pprint
import logging

import torch
import cma
from datetime import datetime

class CMAES:
    def __init__(self, initial_state_dict):
        self.data_queue = list()
        self.performances_queue = list()
        self.default_sigma = 0.5
        self.initial_state_dict = initial_state_dict
        state_flattened = self._flatten_model_state(initial_state_dict)
        self.num_params = len(state_flattened)
        self.pop_size = self.pop_size(self.num_params)
        initial_vector = [0 for _ in range(self.num_params)]
        self.cmaes = cma.CMAEvolutionStrategy(initial_vector, self.default_sigma)
        self.all_features = set()
        logging.info('Set up CMA for {} params, {} pop size'.format(self.num_params, self.pop_size))

    def tell(self, model_state, feature_descriptor, performance):
        """
        Use the information from the run to update the appropriate CMAME generators
        :param model_state: torch model state dict
        :param feature_descriptor: same as in ME, determines which emitter to update with the relevant info
        :param performance: score from eval
        """
        self.all_features.add(feature_descriptor)
        state_flattened = self._flatten_model_state(model_state)

        self.data_queue.append(state_flattened)
        logging.info('added to cmame queue, %d for %s',
                     len(self.data_queue))

        performance = performance.item() if torch.is_tensor(performance) else performance
        # cma is set to minimize, so we invert
        self.performances_queue.append(performance * -1)
        if len(self.data_queue) == self.pop_size:
            logging.info('Asking to prep')
            self.cmaes.ask(number=1)
            logging.info('TELLING FEATURES, lower is better')
            start = datetime.now()
            self.cmaes.tell(self.data_queue, self.performances_queue)
            logging.info('Told performances in %s.', str(datetime.now() - start))
            self.data_queue.clear()
            self.performances_queue.clear()
            logging.info('cleared queue')
            logging.info('current features: %s', pformat(self.all_features))
        logging.debug('recorded perfs to be told')
        logging.debug(pformat(self.performances_queue))


    def ask(self):
        """
        :return: a sample network for evaluation
        """
        logging.info('ASK for next network')
        flattened_state = self.cmaes.ask(number=1)
        model = self._model_state(flattened_state[0].tolist())
        return model

    @staticmethod
    def _flatten_model_state(model_state):
        flattened = torch.cat([a.flatten() for a in model_state.values()])
        return flattened.tolist()

    def _model_state(self, flattened_state):
        new_model = deepcopy(self.initial_state_dict)
        for layer, old_tensor in new_model.items():
            # pop necessary items
            num_els = old_tensor.numel()
            next_els = flattened_state[:num_els]
            del flattened_state[:num_els]

            # set to state
            new_model[layer] = torch.tensor(next_els).view(old_tensor.shape)
        assert 0 == len(flattened_state)
        return new_model

    @staticmethod
    def pop_size(num_params):
        return 4 + int(3 * log(num_params))
