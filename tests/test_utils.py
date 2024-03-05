import warnings
import numpy as np
import pytest
from scipy import special as sp
from numpy.testing import assert_array_equal
from anesthetic import read_chains
from pytest import approx
from anesthetic.utils import (nest_level, compute_nlive, unique, is_int,
                              iso_probability_contours,
                              iso_probability_contours_from_samples,
                              logsumexp, sample_compression_1d,
                              triangular_sample_compression_2d,
                              insertion_p_value, compress_weights,
                              neff)


def test_compress_weights():
    w = compress_weights(w=np.ones(10), u=None)
    assert_array_equal(w, np.ones(10))
    w = compress_weights(w=None, u=np.random.rand(10))
    assert_array_equal(w, np.ones(10))
    r = np.random.rand(10)
    w = compress_weights(w=r, u=None, ncompress=False)
    assert_array_equal(w, r)

    # TODO Remove in 2.1
    with pytest.raises(ValueError):
        compress_weights(w=r, ncompress=-1)


def test_nest_level():
    assert nest_level(0) == 0
    assert nest_level([]) == 1
    assert nest_level(['a']) == 1
    assert nest_level(['a', 'b']) == 1
    assert nest_level([['a'], 'b']) == 2
    assert nest_level(['a', ['b']]) == 2
    assert nest_level([['a'], ['b']]) == 2


def test_compute_nlive():
    # Generate a 'pure' nested sampling run
    np.random.seed(0)
    nlive = 500
    ncompress = 100
    logL = np.cumsum(np.random.rand(nlive, ncompress), axis=1)
    logL_birth = np.concatenate((np.ones((nlive, 1))*-np.inf, logL[:, :-1]),
                                axis=1)
    i = np.argsort(logL.flatten())
    logL = logL.flatten()[i]
    logL_birth = logL_birth.flatten()[i]

    # Compute nlive
    nlives = compute_nlive(logL, logL_birth)

    # Check the first half are constant
    assert_array_equal(nlives[:len(nlives)//2], nlive)

    # Check one point at the end
    assert nlives[-1] == 1

    # Check never more than nlive
    assert nlives.max() <= nlive

    # Check length
    assert (len(nlives) == len(logL))


def test_unique():
    assert unique([3, 2, 1, 4, 1, 3]) == [3, 2, 1, 4]


@pytest.mark.parametrize('ipc', [iso_probability_contours,
                                 iso_probability_contours_from_samples])
def test_iso_probability_contours(ipc):
    p = np.random.randn(10)
    ipc(p)
    with pytest.raises(ValueError):
        ipc(p, contours=[0.68, 0.95])


def test_triangular_sample_compression_2d():
    np.random.seed(0)
    n = 5000
    x = np.random.rand(n)
    y = np.random.rand(n)
    w = np.random.rand(n)
    cov = np.identity(2)
    tri, W = triangular_sample_compression_2d(x, y, cov, w)
    assert len(W) == 1000
    assert sum(W) == pytest.approx(sum(w), rel=1e-1)
    tri, W = triangular_sample_compression_2d(x, y, cov, w, n=False)
    assert len(W) == n
    assert sum(W) == pytest.approx(sum(w))
    tri, W = triangular_sample_compression_2d(x, y, cov, w, n=True)  # entropy
    assert n/2 < len(W) < n
    assert sum(W) == pytest.approx(sum(w), rel=1e-3)
    tri, W = triangular_sample_compression_2d(x, y, cov, w, n='inf')
    assert len(W) == pytest.approx(n/2, rel=1e-1)
    assert sum(W) == pytest.approx(sum(w), rel=1e-2)
    tri, W = triangular_sample_compression_2d(x, y, cov, w, n=10000)
    assert len(W) == n
    assert sum(W) == pytest.approx(sum(w))


def test_sample_compression_1d():
    np.random.seed(0)
    N = 10000
    x_ = np.random.rand(N)
    w_ = np.random.rand(N)
    n = 1000
    x, w = sample_compression_1d(x_, w_, n)
    assert len(x) == n
    assert len(w) == n
    assert np.isclose(w.sum(), w_.sum())

    x_ = np.random.rand(N)
    w_ = np.random.lognormal(1, 0.5, N)
    x, w = sample_compression_1d(x_, w_, ncompress=True)
    assert len(w) <= N


def test_is_int():
    assert is_int(1)
    assert is_int(np.int64(1))
    assert not is_int(1.)
    assert not is_int(np.float64(1.))


def test_logsumexpinf():
    np.random.seed(0)
    a = np.random.rand(10)
    b = np.random.rand(10)
    assert logsumexp(-np.inf, b=[-np.inf]) == -np.inf
    assert logsumexp(a, b=b) == sp.logsumexp(a, b=b)
    a[0] = -np.inf
    assert logsumexp(a, b=b) == sp.logsumexp(a, b=b)
    b[0] = -np.inf
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                'invalid value encountered in multiply',
                                RuntimeWarning)
        assert np.isnan(sp.logsumexp(a, b=b))
    assert np.isfinite(logsumexp(a, b=b))


def test_insertion_p_value():
    np.random.seed(3)
    nlive = 500
    ndead = nlive*20
    indexes = np.random.randint(0, nlive, ndead)
    ks_results = insertion_p_value(indexes, nlive)
    assert 'D' in ks_results
    assert 'p-value' in ks_results
    assert 'sample_size' in ks_results

    assert 'iterations' not in ks_results
    assert 'nbatches' not in ks_results
    assert 'p_value_uncorrected' not in ks_results

    assert ks_results['p-value'] > 0.05
    assert ks_results['sample_size'] == ndead

    ks_results = insertion_p_value(indexes, nlive, 1)
    assert 'D' in ks_results
    assert 'p-value' in ks_results
    assert 'sample_size' in ks_results
    assert 'iterations' in ks_results
    assert 'nbatches' in ks_results
    assert 'uncorrected p-value' in ks_results

    assert ks_results['p-value'] > 0.05
    assert ks_results['uncorrected p-value'] < ks_results['p-value']

    iterations = ks_results['iterations']
    assert isinstance(iterations, tuple)
    assert len(iterations) == 2
    assert iterations[1] - iterations[0] == nlive
    assert ks_results['nbatches'] == 20


def test_p_values_from_sample():
    np.random.seed(3)
    ns = read_chains('./tests/example_data/pc')
    ns._compute_insertion_indexes()
    nlive = len(ns.live_points())

    ks_results = insertion_p_value(ns.insertion[nlive:-nlive], nlive)
    assert ks_results['p-value'] > 0.05

    ks_results = insertion_p_value(ns.insertion[nlive:-nlive], nlive, batch=1)
    assert ks_results['p-value'] > 0.05


@pytest.mark.parametrize('beta', ['entropy', 'kish', 0.5, 1, 2, np.inf,
                                  '0.5', '1', '1.', '1.0', '1.00', 'inf'])
def test_effective_samples(beta):
    w = np.ones(20)
    assert neff(w, beta=beta) == approx(20, rel=1e-6)
    w[10:] = 1e-6
    assert neff(w, beta=beta) == approx(10, rel=1e-2)
    w[15:] = 0
    assert neff(w, beta=beta) == approx(10, rel=1e-2)
