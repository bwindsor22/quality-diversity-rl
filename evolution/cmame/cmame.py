from collections import defaultdict
from math import log
from copy import deepcopy
import random
from pprint import pformat, pprint
import logging

import torch
import cma
from datetime import datetime

class CMAEmitters:
    def __init__(self, initial_state_dict):
        self.emitters = dict() # feature_descriptor -> emitter
        self.data_queue = defaultdict(list) # feature_descriptor -> items to be told to emitter (batch update)
        self.performances_queue = defaultdict(list)
        self.default_sigma = 0.5
        self.initial_state_dict = initial_state_dict
        state_flattened = self._flatten_model_state(initial_state_dict)
        self.num_params = len(state_flattened)
        self.pop_size = self.pop_size(self.num_params)
        logging.info('Set up CMA for {} params, {} pop size'.format(self.num_params, self.pop_size))


    def tell(self, feature_descriptor, model_state,  performance):
        """
        Use the information from the run to update the appropriate CMAME generators
        :param feature_descriptor: same as in ME, determines which emitter to update with the relevant info
        :param model_state: torch model state dict
        :param performance: score from eval
        """
        state_flattened = self._flatten_model_state(model_state)
        if feature_descriptor not in self.emitters:
            initial_vector = [0 for _ in range(self.num_params)]
            cmaes = cma.CMAEvolutionStrategy(initial_vector, self.default_sigma)
            self.emitters[feature_descriptor] = cmaes
            self.emitters[feature_descriptor].ask(number=1)
            logging.info('created new emitter, %s', feature_descriptor)

        self.data_queue[feature_descriptor].append(state_flattened)
        logging.info('added to cmame queue, %d for %s',
                     len(self.data_queue[feature_descriptor]), feature_descriptor)

        # cma is set to minimize, so we invert
        self.performances_queue[feature_descriptor].append(performance.item() * -1)
        if len(self.data_queue[feature_descriptor]) == self.pop_size:
            logging.info('TELLING FEATURES, lower is better')
            logging.info('feature: %s', feature_descriptor)
            logging.info('to tell: %s', str(self.performances_queue[feature_descriptor]))
            start = datetime.now()
            self.emitters[feature_descriptor].tell(self.data_queue[feature_descriptor],
                                                   self.performances_queue[feature_descriptor])
            logging.info('Told features in %s.', str(datetime.now() - start))
            self.data_queue[feature_descriptor].clear()
            self.performances_queue[feature_descriptor].clear()
            logging.info('cleared queue')
        logging.debug('recorded perfs to be told')
        logging.debug(pformat(self.performances_queue))


    def ask(self):
        """
        :return: a sample network for evaluation
        """
        logging.info('ASK for next network among %s', str(list(self.emitters.keys())))
        feature_options = list(self.emitters.keys())
        feature_descriptor = random.choice(feature_options)
        logging.info('feature descriptor %s', feature_descriptor)
        # if model_type not in self.next_eval_queue or not self.next_eval_queue[model_type]:
        #     self.next_eval_queue[model_type] = self.emitters[model_type].ask()
        # flattened_state = self.next_eval_queue[model_type].pop(0)
        flattened_state = self.emitters[feature_descriptor].ask(number=1)
        model = self._model_state(flattened_state[0].tolist())
        logging.info('returning model of type %s', feature_descriptor)
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