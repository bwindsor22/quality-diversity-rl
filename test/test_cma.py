import evolution.cmame.purecma as pcma
import cma
from pprint import pprint
def pure_cma():
    initial = [10, 10]
    es = pcma.CMAES([0, 0], 0.5)
    num_iter = 50
    def to_minimize(x, y):
        return pow(x - 1, 2) + pow(y - 1, 2)

    i = 0
    while i < num_iter:
        i += 1
        X = es.ask()
        pprint(X)
        evals = [to_minimize(e[0], e[1]) for e in X]
        es.tell(X, evals)

def fast_cma():
    es = cma.CMAEvolutionStrategy(2000 * [0], 0.5)
    num_iter = 1000
    i = 0
    while not es.stop() and i < num_iter:
        i += 1
        solutions = es.ask()
        pprint(solutions)
        es.tell(solutions, [cma.ff.rosen(x) for x in solutions])

fast_cma()