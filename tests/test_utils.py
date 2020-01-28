import warnings
import numpy
from numpy.testing import assert_array_equal
from anesthetic.utils import (nest_level, compute_nlive, unique, is_int,
                              triangular_sample_compression_2d, 
                              logsumexp, logsumexpinf)


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

    # Check no points at the end
    assert(nlives[-1] == 0)

    # Check never more than nlive
    assert(nlives.max() <= nlive)


def test_unique():
    assert(unique([3, 2, 1, 4, 1, 3]) == [3, 2, 1, 4])


def test_triangular_sample_compression_2d():
    numpy.random.seed(0)
    n = 5000
    x = numpy.random.rand(n)
    y = numpy.random.rand(n)
    w = numpy.random.rand(n)
    cov = numpy.identity(2)
    tri, W = triangular_sample_compression_2d(x, y, cov, w)
    assert len(W) == 1000
    assert numpy.isclose(sum(W), sum(w), rtol=1e-1)


def test_is_int():
    assert is_int(1)
    assert is_int(numpy.int64(1))
    assert not is_int(1.)
    assert not is_int(numpy.float64(1.))


def test_logsumexpinf():
    a = numpy.random.rand(10)
    b = numpy.random.rand(10)
    assert logsumexpinf(-numpy.inf, b=[-numpy.inf]) == -numpy.inf
    assert logsumexp(a, b=b) == logsumexpinf(a, b=b)
    a[0] = -numpy.inf
    assert logsumexp(a, b=b) == logsumexpinf(a, b=b)
    b[0] = -numpy.inf
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                'invalid value encountered in multiply',
                                RuntimeWarning)
        assert numpy.isnan(logsumexp(a, b=b))
    assert numpy.isfinite(logsumexpinf(a, b=b))
