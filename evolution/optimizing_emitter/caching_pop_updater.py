from math import log
from copy import deepcopy
from evolution.optimizing_emitter.optimizing_emitter import CMAME_Optimizing
from evolution.optimizing_emitter.individual import Individual
import torch
import logging

class CachingPopUpdater:
    """
    Performs the following functions:
    1. Batches CMA updates by storing individuals to tell and generated individuals
    2. Converts models to vectors and vice versa
    3. Logs total counts
    """

    def __init__(self, initial_state_dict):
        self.initial_state_dict = initial_state_dict

        state_flattened = self._flatten_model_state(initial_state_dict)
        self.num_params = len(state_flattened)
        self.pop_size_n = self.pop_size(self.num_params)

        self.ask_cache = []
        self.tell_cache = []

        self.optimizing_emitter = CMAME_Optimizing(state_flattened, 1)

        self.total_count_eval = 0
        self.log_every = 1000

    def ask(self):
        logging.info('Asking for model')
        if not len(self.ask_cache):
            self.ask_cache = self.optimizing_emitter.ask()
        flattened = self.ask_cache.pop(0)
        model_state_dict = self._flattened_to_model_state(flattened)
        logging.info('returning model')
        return model_state_dict

    def tell(self, feature, model_state_dict, fitness):
        self.total_count_eval += 1
        if self.total_count_eval % self.log_every == 0:
            logging.info('LOGGING INTERMEDIATE RESULTS')
            self.optimizing_emitter.feature_map.log()

        flattened = self._flatten_model_state(model_state_dict)
        indiv = Individual(feature, flattened, fitness)
        self.tell_cache.append(indiv)

        if len(self.tell_cache) >= self.pop_size_n:
            self.optimizing_emitter.tell(self.tell_cache)
            self.tell_cache.clear()


    @staticmethod
    def _flatten_model_state(model_state):
        flattened = torch.cat([a.flatten() for a in model_state.values()])
        return flattened.tolist()

    def _flattened_to_model_state(self, flattened_state):
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