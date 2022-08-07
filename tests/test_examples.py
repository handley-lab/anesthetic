import numpy as np
from numpy.testing import assert_allclose
from anesthetic.examples.utils import (
    random_ellipsoid, random_covariance, volume_n_ball, log_volume_n_ball
)
from anesthetic.examples.perfect_ns import (
    gaussian, correlated_gaussian
)
from scipy.spatial import ConvexHull


def test_random_covariance():
    np.random.seed(0)
    d = 5
    sigmas = np.random.rand(d)
    cov = random_covariance(sigmas)
    assert(cov.shape == (d, d))
    e = np.linalg.eigvals(cov)
    assert_allclose(np.sort(e), np.sort(sigmas**2))


def test_random_ellipsoid():
    np.random.seed(0)

    d = 5
    mean = np.random.rand(d)
    cov = random_covariance(np.random.rand(d))

    def rvs(shape=None):
        return random_ellipsoid(mean, cov, shape)

    assert(rvs(10).shape == (10, d))
    assert(rvs((10,)).shape == (10, d))
    assert(rvs((10, 20)).shape == (10, 20, d))

    dat = rvs(100000)

    atol = 1/np.sqrt(len(dat))
    assert_allclose(np.mean(dat, axis=0), mean, atol=atol)
    assert_allclose(np.cov(dat.T), cov/(d+2), atol=atol)

    cov = random_covariance(np.random.rand(2))
    mean = [0, 0]
    dat = rvs(1000000)
    hull = ConvexHull(dat)
    vol = hull.volume
    assert_allclose(vol, np.linalg.det(cov)**0.5*volume_n_ball(2), rtol=1e-3)


def test_perfect_ns_gaussian():
    np.random.seed(0)
    nlive = 500
    ndims = 10
    sigma = 0.1
    R = 1

    samples = gaussian(nlive, ndims, sigma, R)

    assert (samples[:-nlive].nlive == nlive).all()

    mean = samples[np.arange(ndims)].mean()
    assert_allclose(mean, 0, atol=1e-2)
    cov = samples[np.arange(ndims)].cov()
    assert_allclose(cov, np.eye(ndims)*sigma**2, atol=1e-3)

    D = log_volume_n_ball(ndims) - ndims/2*np.log(2*np.pi*np.e*sigma**2)
    assert_allclose(samples.D(), D, atol=3*samples.D(12).std())

    logZ = ndims/2 * np.log(2*np.pi*sigma**2) - log_volume_n_ball(ndims, R)
    assert_allclose(samples.logZ(), logZ, atol=3*samples.logZ(12).std())


def test_perfect_ns_correlated_gaussian():
    np.random.seed(0)
    nlive = 500
    ndims = 5
    mean = 0.5*np.ones(ndims)
    cov = random_covariance(np.random.rand(ndims)*0.1)
    samples = correlated_gaussian(nlive, mean, cov)
    assert_allclose(samples.logZ(), 0, atol=3*samples.logZ(12).std())
    assert (samples[:-nlive].nlive >= nlive).all()
    assert_allclose(samples[np.arange(ndims)].mean(), mean, atol=1e-2)
    assert_allclose(samples[np.arange(ndims)].cov(), cov/(ndims+2), atol=1e-2)
