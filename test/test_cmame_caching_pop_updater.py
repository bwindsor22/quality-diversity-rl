from unittest import TestCase, skip

import torch
from evolution.optimizing_emitter.caching_pop_updater import CachingPopUpdater


evals = 10000

def func_to_try(x):
    x1, x2 = x[0], x[1]
    if abs(x1) > 1000 or abs(x2) > 1000:
        return 0
    return 10 * x1 ** 2 + 10 * x2 ** 2 - x1 ** 4 - x2 ** 4

class CMAMETest(TestCase):

    def test_pop_optimize(self):
        initial_model = {'layer': torch.tensor([0, 0])}
        cpu = CachingPopUpdater(initial_model, log_every=5)
        for _ in range(evals):
            model = cpu.ask()
            x = model['layer'].tolist()
            fitness = torch.tensor(func_to_try(x))
            feature = '{}-{}'.format(int(x[0] > 0), int(x[1] > 0))
            cpu.tell(feature, model, fitness)

        final_map = cpu.optimizing_emitter.cmame_feature_map.elite_map
        # should be > 45 for most quadrants
        print(final_map)