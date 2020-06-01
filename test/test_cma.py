import evolution.reference_multi_cmame.purecma as pcma
from evolution.fast_optimizing_emitter.pycma.cma import CMAEvolutionStrategy
from pprint import pprint

def to_minimize(x, y, z):
    return pow(x - 1, 2) + pow(y - 1, 2) + pow(z - 3, 2)

def pure_cma():
    initial = [10, 10]
    es = pcma.CMAES([0, 0, 0], 0.5)
    num_iter = 50

    i = 0
    while i < num_iter:
        i += 1
        X = es.ask()
        pprint(X)
        evals = [to_minimize(e[0], e[1], e[2]) for e in X]
        es.tell(X, evals)

def fast_cma():
    es = CMAEvolutionStrategy(3 * [10], 0.5)
    num_iter = 50
    i = 0
    pop_size = len(es.ask())
    while not es.stop() and i < num_iter:
        i += 1
        sols = []
        for _ in range(pop_size):
            sols.extend(es.ask(number=1))
        pprint(sols)
        es.tell(sols, [to_minimize(*x) for x in sols])

fast_cma()