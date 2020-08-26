from sys import stdout as _stdout # not strictly necessary
from typing import List
import logging
from math import log, exp
from random import normalvariate as random_normalvariate
from datetime import datetime
from evolution.optimizing_emitter.feature_map import FeatureMap

try:
    from evolution.cmame.interfaces import OOOptimizer, BaseDataLogger as _BaseDataLogger
except (ImportError, ValueError):
    OOOptimizer, _BaseDataLogger = object, object



class CMAESParameters(object):
    """static "internal" parameter setting for `CMAES`

    """
    default_popsize = '4 + int(3 * log(N))'
    def __init__(self, N, popsize=None,
                 RecombinationWeights=None):
        """set static, fixed "strategy" parameters once and for all.

        Input parameter ``RecombinationWeights`` may be set to the class
        `RecombinationWeights`.
        """
        self.dimension = N
        self.chiN = (1 - 1. / (4 * N) + 1. / (21 * N**2))

        # Strategy parameter setting: Selection
        self.lam = eval(safe_str(popsize if popsize else
                                 CMAESParameters.default_popsize,
                                 {'int': 'int', 'log': 'log', 'N': N}))
        self.mu = int(self.lam / 2)  # number of parents/points/solutions for recombination
        if RecombinationWeights:
            self.weights = RecombinationWeights(self.lam)
            self.mueff = self.weights.mueff
        else:  # set non-negative recombination weights "manually"
            _weights = [log(self.lam / 2 + 0.5) - log(i + 1) if i < self.mu else 0
                        for i in range(self.lam)]
            w_sum = sum(_weights[:self.mu])
            self.weights = [w / w_sum for w in _weights]  # sum is one now
            self.mueff = sum(self.weights[:self.mu])**2 / \
                         sum(w**2 for w in self.weights[:self.mu])  # variance-effectiveness of sum w_i x_i

        # Strategy parameter setting: Adaptation
        self.cc = (4 + self.mueff/N) / (N+4 + 2 * self.mueff/N)  # time constant for cumulation for C
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)  # time constant for cumulation for sigma control
        self.c1 = 2 / ((N + 1.3)**2 + self.mueff)  # learning rate for rank-one update of C
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((N + 2)**2 + self.mueff)])  # and for rank-mu update
        self.damps = 2 * self.mueff/self.lam + 0.3 + self.cs  # damping for sigma, usually close to 1

        if RecombinationWeights:
            self.weights.finalize_negative_weights(N, self.c1, self.cmu)
        # gap to postpone eigendecomposition to achieve O(N**2) per eval
        # 0.5 is chosen such that eig takes 2 times the time of tell in >=20-D
        self.lazy_gap_evals = 0.5 * N * self.lam * (self.c1 + self.cmu)**-1 / N**2

class CMAME_Optimizing(OOOptimizer):  # could also inherit from object
    """class for non-linear non-convex numerical minimization with CMA-ES.

    The class implements the interface define in `OOOptimizer`, namely
    the methods `__init__`, `ask`, `tell`, `stop`, `disp` and property
    `result`.

    Examples
    --------

    The Jupyter notebook or IPython are the favorite environments to
    execute these examples, both in ``%pylab`` mode. All examples
    minimize the function `elli`, output is not shown.

    First we need to import the module we want to use. We import `purecma`
    from `cma` as (aliased to) ``pcma``::

        from cma import purecma as pcma

    The shortest example uses the inherited method
    `OOOptimizer.optimize`::

        es = pcma.CMAES(8 * [0.1], 0.5).optimize(pcma.ff.elli)

    See method `CMAES.__init__` for a documentation of the input
    parameters to `CMAES`. We might have a look at the result::

        print(es.result[0])  # best solution and
        print(es.result[1])  # its function value

    `result` is a property of `CMAES`. In order to display more exciting
    output, we may use the `CMAESDataLogger` instance in the `logger`
    attribute of `CMAES`::

        es.logger.plot()  # if matplotlib is available

    Virtually the same example can be written with an explicit loop
    instead of using `optimize`, see also `fmin`. This gives insight
    into the `CMAES` class interface and entire control over the
    iteration loop::

        pcma.fmin??  # print source, works in jupyter/ipython only
        es = pcma.CMAES(9 * [0.5], 0.3)  # calls CMAES.__init__()

        # this loop resembles the method optimize
        while not es.stop():  # iterate
            X = es.ask()      # get candidate solutions
            f = [pcma.ff.elli(x) for x in X]  # evaluate solutions
            es.tell(X, f)     # do all the real work
            es.disp(20)       # display info every 20th iteration
            es.logger.add(es) # log another "data line"

        # final output
        print('termination by', es.stop())
        print('best f-value =', es.result[1])
        print('best solution =', es.result[0])

        print('potentially better solution xmean =', es.result[5])
        print("let's check f(xmean) = ", pcma.ff.elli(es.result[5]))
        es.logger.plot()  # if matplotlib is available

    A very similar example which may also save the logged data within
    the loop is the implementation of function `fmin`.

    Details
    -------
    Most of the work is done in the method `tell`. The property
    `result` contains more useful output.

    :See: `fmin`, `OOOptimizer.optimize`

    """
    def __init__(self, xstart: List, sigma,  # mandatory
                 popsize=CMAESParameters.default_popsize,
                 ftarget=None,
                 maxfevals='100 * popsize + '  # 100 iterations plus...
                           '150 * (N + 3)**2 * popsize**0.5',
                 randn=random_normalvariate):
        """Instantiate `CMAES` object instance using `xstart` and `sigma`.

        Parameters
        ----------
            `xstart`: `list`
                of numbers (like ``[3, 2, 1.2]``), initial
                solution vector
            `sigma`: `float`
                initial step-size (standard deviation in each coordinate)
            `popsize`: `int` or `str`
                population size, number of candidate samples per iteration
            `maxfevals`: `int` or `str`
                maximal number of function evaluations, a string is
                evaluated with ``N`` as search space dimension
            `ftarget`: `float`
                target function value
            `randn`: `callable`
                normal random number generator, by default
                `random.normalvariate`

        Details: this method initializes the dynamic state variables and
        creates a `CMAESParameters` instance for static parameters.
        """
        # process some input parameters and set static parameters
        self.N = len(xstart)  # number of objective variables/problem dimension
        self.params = CMAESParameters(self.N, popsize)
        self.maxfevals = eval(safe_str(maxfevals,
                                       known_words={'N': self.N, 'popsize': self.params.lam}))
        self.ftarget = ftarget  # stop if fitness <= ftarget
        self.randn = randn
        self.sigma_start = sigma

        # initializing dynamic state variables
        self.xmean = xstart[:] # initial point, distribution mean, a copy
        self.sigma = sigma
        self.pc = self.N * [0]  # evolution path for C
        self.ps = self.N * [0]  # and for sigma
        self.C = DecomposingPositiveMatrix(self.N)  # covariance matrix
        self.counteval = 0  # countiter should be equal to counteval / lam
        self.fitvals = []   # for bookkeeping output and termination

        self.feature_map = FeatureMap()


    def ask(self):
        """sample lambda candidate solutions

        distributed according to::

            m + sigma * Normal(0,C) = m + sigma * B * D * Normal(0,I)
                                    = m + B * D * sigma * Normal(0,I)

        and return a `list` of the sampled "vectors".
        """
        self.C.update_eigensystem(self.counteval,
                                  self.params.lazy_gap_evals)
        candidate_solutions = []
        start = datetime.now()
        for k in range(self.params.lam):  # repeat lam times
            logging.info('generating at %d of %d', k, self.params.lam)
            z = [self.sigma * eigenval**0.5 * self.randn(0, 1)
                 for eigenval in self.C.eigenvalues]
            y = dot(self.C.eigenbasis, z)
            candidate_solutions.append(plus(self.xmean, y))
        logging.info('Generated solutions in {}'.format(str(datetime.now() - start)))
        return candidate_solutions

    def tell(self, individuals):
        """update the evolution paths and the distribution parameters m,
        sigma, and C within CMA-ES.

        Parameters
        ----------
            `arx`: `list` of "row vectors"
                a list of candidate solution vectors, presumably from
                calling `ask`. ``arx[k][i]`` is the i-th element of
                solution vector k.
            `fitvals`: `list`
                the corresponding objective function values, to be
                minimised
        """
        logging.info('Tellling individuals...')
        self.feature_map.add_individuals_to_map(individuals)

        arx =  [indiv.flattened_state_dict for indiv in individuals]
        # CMAME is set to minimize
        fitvals = [indiv.fitness.item() * -1 for indiv in individuals]

        ### bookkeeping and convenience short cuts
        self.counteval += len(fitvals)  # evaluations used within tell
        N = len(self.xmean)
        par = self.params
        xold = self.xmean  # not a copy, xmean is assigned anew later

        ### Sort by fitness
        arx = [arx[k] for k in argsort(fitvals)]  # sorted arx
        self.fitvals = sorted(fitvals)  # used for termination and display only

        ### recombination, compute new weighted mean value
        self.xmean = dot(arx[0:par.mu], par.weights[:par.mu], transpose=True)
        #          = [sum(self.weights[k] * arx[k][i] for k in range(self.mu))
        #                                             for i in range(N)]

        ### Cumulation: update evolution paths
        y = minus(self.xmean, xold)
        z = dot(self.C.invsqrt, y)  # == C**(-1/2) * (xnew - xold)
        csn = (par.cs * (2 - par.cs) * par.mueff)**0.5 / self.sigma
        for i in range(N):  # update evolution path ps
            self.ps[i] = (1 - par.cs) * self.ps[i] + csn * z[i]
        ccn = (par.cc * (2 - par.cc) * par.mueff)**0.5 / self.sigma
        # turn off rank-one accumulation when sigma increases quickly
        hsig = (sum(x**2 for x in self.ps) / N  # ||ps||^2 / N is 1 in expectation
                / (1-(1-par.cs)**(2*self.counteval/par.lam))  # account for initial value of ps
                < 2 + 4./(N+1))  # should be smaller than 2 + ...
        for i in range(N):  # update evolution path pc
            self.pc[i] = (1 - par.cc) * self.pc[i] + ccn * hsig * y[i]

        ### Adapt covariance matrix C
        # minor adjustment for the variance loss from hsig
        c1a = par.c1 * (1 - (1-hsig**2) * par.cc * (2-par.cc))
        self.C.multiply_with(1 - c1a - par.cmu * sum(par.weights))  # C *= 1 - c1 - cmu * sum(w)
        self.C.addouter(self.pc, par.c1)  # C += c1 * pc * pc^T, so-called rank-one update
        for k, wk in enumerate(par.weights):  # so-called rank-mu update
            if wk < 0:  # guaranty positive definiteness
                wk *= N * (self.sigma / self.C.mahalanobis_norm(minus(arx[k], xold)))**2
            self.C.addouter(minus(arx[k], xold),  # C += wk * cmu * dx * dx^T
                            wk * par.cmu / self.sigma**2)

        ### Adapt step-size sigma
        cn, sum_square_ps = par.cs / par.damps, sum(x**2 for x in self.ps)
        # "mutation power" in C#
        self.sigma *= exp(min(1, cn * (sum_square_ps / N - 1) / 2))
        # self.sigma *= exp(min(1, cn * (sum_square_ps**0.5 / par.chiN - 1)))

        ## needs restart
        if self.needs_reset(fitvals):
            logging.info('Resetting CMA-ES at {} evals', self.counteval)
            self.reset()

        logging.info('Finished tell. %d evaluated in this cell', self.counteval)

    def needs_reset(self, fitvals):
        min_fit = min(fitvals)
        max_fit = max(fitvals)
        logging.info('testing reset: max: %d, min: %d', max_fit, min_fit)
        if max_fit - min_fit < 1:
            logging.info('Reset CMA-ES because of flatness')
            return True

        area = self.sigma * (abs(max(self.C.eigenvalues)) ** 0.5)
        logging.info('testing reset: area: %d', area * 1e11)
        if area < 1e-11:
            logging.info('Reset CMA-ES because of area')
            return True

        return False


    def reset(self):
        if self.feature_map.count_elites() > 0:
            individual = self.feature_map.get_random_elite()
            self.xmean = individual.param_vector
        else:
            self.xmean = [self.randn(0, self.sigma_start) for _ in range(self.N)]

        self.C = DecomposingPositiveMatrix(self.N)  # covariance matrix
        self.counteval = 0

        self.pc = self.N * [0]  # evolution path for C
        self.ps = self.N * [0]  # and for sigma
        self.sigma = self.sigma_start



    def stop(self):
        """return satisfied termination conditions in a dictionary,

        generally speaking like ``{'termination_reason':value, ...}``,
        for example ``{'tolfun':1e-12}``, or the empty `dict` ``{}``.
        """
        res = {}
        if self.counteval <= 0:
            return res
        if self.counteval >= self.maxfevals:
            res['maxfevals'] = self.maxfevals
        if self.ftarget is not None and len(self.fitvals) > 0 \
                and self.fitvals[0] <= self.ftarget:
            res['ftarget'] = self.ftarget
        if self.C.condition_number > 1e14:
            res['condition'] = self.C.condition_number
        if len(self.fitvals) > 1 \
                and self.fitvals[-1] - self.fitvals[0] < 1e-12:
            res['tolfun'] = 1e-12
        if self.sigma * max(self.C.eigenvalues)**0.5 < 1e-11:
            # remark: max(D) >= max(diag(C))**0.5
            res['tolx'] = 1e-11
        return res

    @property
    def result(self):
        """the `tuple` ``(xbest, f(xbest), evaluations_xbest, evaluations,
        iterations, xmean, stds)``
        """
        return (self.counteval,
                int(self.counteval / self.params.lam),
                self.xmean,
                [self.sigma * C_ii**0.5 for C_ii in self.C.diag])

    def disp(self, verb_modulo=1):
        """`print` some iteration info to `stdout`
        """
        if verb_modulo is None:
            verb_modulo = 20
        if not verb_modulo:
            return
        iteration = self.counteval / self.params.lam

        if iteration == 1 or iteration % (10 * verb_modulo) < 1:
            print('evals: ax-ratio max(std)   f-value')
        if iteration <= 2 or iteration % verb_modulo < 1:
            print(str(self.counteval).rjust(5) + ': ' +
                  ' %6.1f %8.1e  ' % (self.C.condition_number**0.5,
                                      self.sigma * max(self.C.diag)**0.5) +
                  str(self.fitvals[0]))
            _stdout.flush()


#_____________________________________________________________________
#_______________________ Helper Class&Functions ______________________
#
class BestSolution(object):
    """container to keep track of the best solution seen"""
    def __init__(self, x=None, f=None, evals=None):
        """take `x`, `f`, and `evals` to initialize the best solution
        """
        self.x, self.f, self.evals = x, f, evals

    def update(self, x, f, evals=None):
        """update the best solution if ``f < self.f``
        """
        if self.f is None or f < self.f:
            self.x = x
            self.f = f
            self.evals = evals
        return self
    @property
    def all(self):
        """``(x, f, evals)`` of the best seen solution"""
        return self.x, self.f, self.evals

class SquareMatrix(list):  # inheritance from numpy.ndarray is not recommended
    """rudimental square matrix class"""
    def __init__(self, dimension):
        """initialize with identity matrix"""
        for i in range(dimension):
            self.append(dimension * [0])
            self[i][i] = 1

    def multiply_with(self, factor):
        """multiply matrix in place with `factor`"""
        for row in self:
            for j in range(len(row)):
                row[j] *= factor
        return self

    def addouter(self, b, factor=1):
        """Add in place `factor` times outer product of vector `b`,

        without any dimensional consistency checks.
        """
        for i, row in enumerate(self):
            for j in range(len(row)):
                row[j] += factor * b[i] * b[j]
        return self
    @property
    def diag(self):
        """diagonal of the matrix as a copy (save to change)
        """
        return [self[i][i] for i in range(len(self)) if i < len(self[i])]

class DecomposingPositiveMatrix(SquareMatrix):
    """Symmetric matrix maintaining its own eigendecomposition.

    If ``isinstance(C, DecomposingPositiveMatrix)``,
    the eigendecomposion (the return value of `eig`) is stored in
    the attributes `eigenbasis` and `eigenvalues` such that the i-th
    eigenvector is::

        [row[i] for row in C.eigenbasis]  # or equivalently
        [C.eigenbasis[j][i] for j in range(len(C.eigenbasis))]

    with eigenvalue ``C.eigenvalues[i]`` and hence::

        C = C.eigenbasis x diag(C.eigenvalues) x C.eigenbasis^T

    """
    def __init__(self, dimension):
        SquareMatrix.__init__(self, dimension)
        self.eigenbasis = eye(dimension)
        self.eigenvalues = dimension * [1]
        self.condition_number = 1
        self.invsqrt = eye(dimension)
        self.updated_eval = 0

    def update_eigensystem(self, current_eval, lazy_gap_evals):
        """Execute eigendecomposition of `self` if
        ``current_eval > lazy_gap_evals + last_updated_eval``.

        Assumes (for sake of simplicity) that `self` is positive
        definite and hence raises a `RuntimeError` otherwise.
        """
        if current_eval <= self.updated_eval + lazy_gap_evals:
            return self
        self._enforce_symmetry()  # probably not necessary with eig
        self.eigenvalues, self.eigenbasis = eig(self)  # O(N**3)
        if min(self.eigenvalues) <= 0:
            raise RuntimeError(
                "The smallest eigenvalue is <= 0 after %d evaluations!"
                "\neigenvectors:\n%s \neigenvalues:\n%s"
                % (current_eval, str(self.eigenbasis), str(self.eigenvalues)))
        self.condition_number = max(self.eigenvalues) / min(self.eigenvalues)
        # now compute invsqrt(C) = C**(-1/2) = B D**(-1/2) B'
        # this is O(n^3) and takes about 25% of the time of eig
        for i in range(len(self)):
            for j in range(i+1):
                self.invsqrt[i][j] = self.invsqrt[j][i] = sum(
                    self.eigenbasis[i][k] * self.eigenbasis[j][k]
                    / self.eigenvalues[k]**0.5 for k in range(len(self)))
        self.updated_eval = current_eval
        return self

    def mahalanobis_norm(self, dx):
        """return ``(dx^T * C^-1 * dx)**0.5``
        """
        return sum(xi**2 for xi in dot(self.invsqrt, dx))**0.5

    def _enforce_symmetry(self):
        for i in range(len(self)):
            for j in range(i):
                self[i][j] = self[j][i] = (self[i][j] + self[j][i]) / 2
        return self

def eye(dimension):
    """return identity matrix as `list` of "vectors" (lists themselves)"""
    m = [dimension * [0] for i in range(dimension)]
    # m = N * [N * [0]] fails because it gives N times the same reference
    for i in range(dimension):
        m[i][i] = 1
    return m

def dot(A, b, transpose=False):
    """ usual dot product of "matrix" A with "vector" b.

    ``A[i]`` is the i-th row of A. With ``transpose=True``, A transposed
    is used.
    """
    if not transpose:
        return [sum(A[i][j] * b[j] for j in range(len(b)))
                for i in range(len(A))]
    else:
        return [sum(A[j][i] * b[j] for j in range(len(b)))
                for i in range(len(A[0]))]

def plus(a, b):
    """add vectors, return a + b """
    return [a[i] + b[i] for i in range(len(a))]

def minus(a, b):
    """subtract vectors, return a - b"""
    return [a[i] - b[i] for i in range(len(a))]

def argsort(a):
    """return index list to get `a` in order, ie
    ``a[argsort(a)[i]] == sorted(a)[i]``
    """
    return sorted(range(len(a)), key=a.__getitem__)  # a.__getitem__(i) is a[i]

def safe_str(s, known_words=None):
    """return ``s`` as `str` safe to `eval` or raise an exception.

    Strings in the `dict` `known_words` are replaced by their values
    surrounded with a space, which the caller considers safe to evaluate
    with `eval` afterwards.

    Known issues:

    >>> try: from cma.purecma import safe_str
    ... except ImportError: from purecma import safe_str
    >>> safe_str('int(p)', {'int': 'int', 'p': 3.1})  # fine
    ' int ( 3.1 )'
    >>> safe_str('int(n)', {'int': 'int', 'n': 3.1})  # unexpected
    ' i 3.1 t ( 3.1 )'

    """
    safe_chars = ' 0123456789.,+-*()[]e'
    if s != str(s):
        return str(s)
    if not known_words:
        known_words = {}
    stest = s[:]  # test this string
    sret = s[:]  # return this string
    for word in sorted(known_words.keys(), key=len, reverse=True):
        stest = stest.replace(word, '  ')
        sret = sret.replace(word, " %s " % known_words[word])
    for c in stest:
        if c not in safe_chars:
            raise ValueError('"%s" is not a safe string'
                             ' (known words are %s)' % (s, str(known_words)))
    return sret

#____________________________________________________________
#____________________________________________________________
#
# C and B are arrays rather than matrices, because they are
# addressed via B[i][j], matrices can only be addressed via B[i,j]

# tred2(N, B, diagD, offdiag);
# tql2(N, diagD, offdiag, B);

# Symmetric Householder reduction to tridiagonal form, translated from
#   JAMA package.

def \
        eig(C):
    """eigendecomposition of a symmetric matrix.

    Return the eigenvalues and an orthonormal basis
    of the corresponding eigenvectors, ``(EVals, Basis)``, where

    - ``Basis[i]``: `list`, is the i-th row of ``Basis``
    - the i-th column of ``Basis``, ie ``[Basis[j][i] for j in range(len(Basis))]``
      is the i-th eigenvector with eigenvalue ``EVals[i]``

    Details: much slower than `numpy.linalg.eigh`.
    """
    # class eig(object):
    #     def __call__(self, C):

    # Householder transformation of a symmetric matrix V into tridiagonal
    #   form.
    # -> n             : dimension
    # -> V             : symmetric nxn-matrix
    # <- V             : orthogonal transformation matrix:
    #                    tridiag matrix == V * V_in * V^t
    # <- d             : diagonal
    # <- e[0..n-1]     : off diagonal (elements 1..n-1)

    # Symmetric tridiagonal QL algorithm, iterative
    # Computes the eigensystem from a tridiagonal matrix in roughtly 3N^3
    #    operations
    # -> n     : Dimension.
    # -> d     : Diagonale of tridiagonal matrix.
    # -> e[1..n-1] : off-diagonal, output from Householder
    # -> V     : matrix output von Householder
    # <- d     : eigenvalues
    # <- e     : garbage?
    # <- V     : basis of eigenvectors, according to d

    #  tred2(N, B, diagD, offdiag); B=C on input
    #  tql2(N, diagD, offdiag, B);

    #import numpy as np
    #return np.linalg.eigh(C)  # return sorted EVs
    try:
        num_opt = False  # True doesn't work (yet)
        if num_opt:
            import numpy as np
    except ImportError:
        num_opt = False

    #  private void tred2 (int n, double V[][], double d[], double e[]) {
    def tred2(n, V, d, e):
        #  This is derived from the Algol procedures tred2 by
        #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        #  Fortran subroutine in EISPACK.

        # num_opt = False  # factor 1.5 in 30-D

        d[:] = V[n-1][:]  # d is output argument
        if num_opt:
            # V = np.asarray(V, dtype=float)
            e = np.asarray(e, dtype=float)

        # Householder reduction to tridiagonal form.

        for i in range(n-1, 0, -1):
            # Scale to avoid under/overflow.
            h = 0.0
            if not num_opt:
                scale = 0.0
                for k in range(i):
                    scale = scale + abs(d[k])
            else:
                scale = sum(np.abs(d[0:i]))

            if scale == 0.0:
                e[i] = d[i-1]
                for j in range(i):
                    d[j] = V[i-1][j]
                    V[i][j] = 0.0
                    V[j][i] = 0.0
            else:

                # Generate Householder vector.
                if not num_opt:
                    for k in range(i):
                        d[k] /= scale
                        h += d[k] * d[k]
                else:
                    d[:i] /= scale
                    h = np.dot(d[:i], d[:i])

                f = d[i-1]
                g = h**0.5

                if f > 0:
                    g = -g

                e[i] = scale * g
                h -= f * g
                d[i-1] = f - g
                if not num_opt:
                    for j in range(i):
                        e[j] = 0.0
                else:
                    e[:i] = 0.0

                # Apply similarity transformation to remaining columns.
                for j in range(i):
                    f = d[j]
                    V[j][i] = f
                    g = e[j] + V[j][j] * f
                    if not num_opt:
                        for k in range(j+1, i):
                            g += V[k][j] * d[k]
                            e[k] += V[k][j] * f
                        e[j] = g
                    else:
                        e[j+1:i] += V.T[j][j+1:i] * f
                        e[j] = g + np.dot(V.T[j][j+1:i], d[j+1:i])

                f = 0.0
                if not num_opt:
                    for j in range(i):
                        e[j] /= h
                        f += e[j] * d[j]
                else:
                    e[:i] /= h
                    f += np.dot(e[:i], d[:i])

                hh = f / (h + h)
                if not num_opt:
                    for j in range(i):
                        e[j] -= hh * d[j]
                else:
                    e[:i] -= hh * d[:i]

                for j in range(i):
                    f = d[j]
                    g = e[j]
                    if not num_opt:
                        for k in range(j, i):
                            V[k][j] -= (f * e[k] + g * d[k])
                    else:
                        V.T[j][j:i] -= (f * e[j:i] + g * d[j:i])

                    d[j] = V[i-1][j]
                    V[i][j] = 0.0

            d[i] = h
        # end for i--

        # Accumulate transformations.

        for i in range(n-1):
            V[n-1][i] = V[i][i]
            V[i][i] = 1.0
            h = d[i+1]
            if h != 0.0:
                if not num_opt:
                    for k in range(i+1):
                        d[k] = V[k][i+1] / h
                else:
                    d[:i+1] = V.T[i+1][:i+1] / h

                for j in range(i+1):
                    if not num_opt:
                        g = 0.0
                        for k in range(i+1):
                            g += V[k][i+1] * V[k][j]
                        for k in range(i+1):
                            V[k][j] -= g * d[k]
                    else:
                        g = np.dot(V.T[i+1][0:i+1], V.T[j][0:i+1])
                        V.T[j][:i+1] -= g * d[:i+1]

            if not num_opt:
                for k in range(i+1):
                    V[k][i+1] = 0.0
            else:
                V.T[i+1][:i+1] = 0.0

        if not num_opt:
            for j in range(n):
                d[j] = V[n-1][j]
                V[n-1][j] = 0.0
        else:
            d[:n] = V[n-1][:n]
            V[n-1][:n] = 0.0

        V[n-1][n-1] = 1.0
        e[0] = 0.0

    # Symmetric tridiagonal QL algorithm, taken from JAMA package.
    # private void tql2 (int n, double d[], double e[], double V[][]) {
    # needs roughly 3N^3 operations
    def tql2(n, d, e, V):
        #  This is derived from the Algol procedures tql2, by
        #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        #  Fortran subroutine in EISPACK.

        # num_opt = False  # True doesn't work

        if not num_opt:
            for i in range(1, n):  # (int i = 1; i < n; i++):
                e[i-1] = e[i]
        else:
            e[0:n-1] = e[1:n]
        e[n-1] = 0.0

        f = 0.0
        tst1 = 0.0
        eps = 2.0**-52.0
        for l in range(n):  # (int l = 0; l < n; l++) {

            # Find small subdiagonal element

            tst1 = max(tst1, abs(d[l]) + abs(e[l]))
            m = l
            while m < n:
                if abs(e[m]) <= eps*tst1:
                    break
                m += 1

            # If m == l, d[l] is an eigenvalue,
            # otherwise, iterate.

            if m > l:
                iiter = 0
                while 1:  # do {
                    iiter += 1  # (Could check iteration count here.)

                    # Compute implicit shift

                    g = d[l]
                    p = (d[l+1] - g) / (2.0 * e[l])
                    r = (p**2 + 1)**0.5  # hypot(p, 1.0)
                    if p < 0:
                        r = -r

                    d[l] = e[l] / (p + r)
                    d[l+1] = e[l] * (p + r)
                    dl1 = d[l+1]
                    h = g - d[l]
                    if not num_opt:
                        for i in range(l+2, n):
                            d[i] -= h
                    else:
                        d[l+2:n] -= h

                    f = f + h

                    # Implicit QL transformation.

                    p = d[m]
                    c = 1.0
                    c2 = c
                    c3 = c
                    el1 = e[l+1]
                    s = 0.0
                    s2 = 0.0

                    # hh = V.T[0].copy()  # only with num_opt
                    for i in range(m-1, l-1, -1):
                        # (int i = m-1; i >= l; i--) {
                        c3 = c2
                        c2 = c
                        s2 = s
                        g = c * e[i]
                        h = c * p
                        r = (p**2 + e[i]**2)**0.5  # hypot(p,e[i])
                        e[i+1] = s * r
                        s = e[i] / r
                        c = p / r
                        p = c * d[i] - s * g
                        d[i+1] = h + s * (c * g + s * d[i])

                        # Accumulate transformation.

                        if not num_opt:  # overall factor 3 in 30-D
                            for k in range(n):  # (int k = 0; k < n; k++){
                                h = V[k][i+1]
                                V[k][i+1] = s * V[k][i] + c * h
                                V[k][i] = c * V[k][i] - s * h
                        else:  # about 20% faster in 10-D
                            hh = V.T[i+1].copy()
                            # hh[:] = V.T[i+1][:]
                            V.T[i+1] = s * V.T[i] + c * hh
                            V.T[i] = c * V.T[i] - s * hh
                            # V.T[i] *= c
                            # V.T[i] -= s * hh

                    p = -s * s2 * c3 * el1 * e[l] / dl1
                    e[l] = s * p
                    d[l] = c * p

                    # Check for convergence.
                    if abs(e[l]) <= eps*tst1:
                        break
                # } while (Math.abs(e[l]) > eps*tst1);

            d[l] += f
            e[l] = 0.0

        # Sort eigenvalues and corresponding vectors.
        if 11 < 3:
            for i in range(n-1):  # (int i = 0; i < n-1; i++) {
                k = i
                p = d[i]
                for j in range(i+1, n):  # (int j = i+1; j < n; j++) {
                    if d[j] < p:  # NH find smallest k>i
                        k = j
                        p = d[j]

                if k != i:
                    d[k] = d[i]  # swap k and i
                    d[i] = p
                    for j in range(n):  # (int j = 0; j < n; j++) {
                        p = V[j][i]
                        V[j][i] = V[j][k]
                        V[j][k] = p
    # tql2
    N = len(C[0])
    V = [C[i][:] for i in range(N)]
    d = N * [0]
    e = N * [0]
    tred2(N, V, d, e)
    tql2(N, d, e, V)
    return d, V  # sorting of V-columns in place is non-trivial