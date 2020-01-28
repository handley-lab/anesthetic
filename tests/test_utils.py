import numpy
from numpy.testing import assert_array_equal
from anesthetic.utils import nest_level, compute_nlive, unique
from anesthetic.utils import logsumexp, logsumexpinf


def test_nest_level():
    assert(nest_level(0) == 0)
    assert(nest_level([]) == 1)
    assert(nest_level(['a']) == 1)
    assert(nest_level(['a', 'b']) == 1)
    assert(nest_level([['a'], 'b']) == 2)
    assert(nest_level(['a', ['b']]) == 2)
    assert(nest_level([['a'], ['b']]) == 2)


def test_compute_nlive():
    # Generate a 'pure' nested sampling run
    numpy.random.seed(0)
    nlive = 500
    ncompress = 100
    logL = numpy.cumsum(numpy.random.rand(nlive, ncompress), axis=1)
    logL_birth = numpy.concatenate((numpy.ones((nlive, 1))*-1e30,
                                    logL[:, :-1]), axis=1)
    i = numpy.argsort(logL.flatten())
    logL = logL.flatten()[i]
    logL_birth = logL_birth.flatten()[i]

    # Compute nlive
    nlives = compute_nlive(logL, logL_birth)

    # Check the first half are constant
    assert_array_equal(nlives[:len(nlives)//2], nlive)

    # Check one points at the end
    assert(nlives[-1] == 1)

    # Check never more than nlive
    assert(nlives.max() <= nlive)


def test_unique():
    assert(unique([3, 2, 1, 4, 1, 3]) == [3, 2, 1, 4])


def test_logsumexpinf():
    a = numpy.random.rand(10)
    b = numpy.random.rand(10)
    assert logsumexpinf(-numpy.inf, b=[-numpy.inf]) == -numpy.inf
    assert logsumexp(a, b=b) == logsumexpinf(a, b=b)
    a[0] = -numpy.inf
    assert logsumexp(a, b=b) == logsumexpinf(a, b=b)
    b[0] = -numpy.inf
    assert numpy.isnan(logsumexp(a, b=b))
    assert numpy.isfinite(logsumexpinf(a, b=b))
