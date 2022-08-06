import numpy as np
from numpy.testing import assert_allclose
from anesthetic.examples.utils import (
    random_ellipsoid, random_covariance, volume_n_ball, log_volume_n_ball
)
from anesthetic.examples.perfect_ns import (
    gaussian, correlated_gaussian
)
from scipy.spatial import ConvexHull
from scipy.special import gamma, gammaln


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
    D = ndims/2*np.log(2*np.pi*np.e*sigma**2) - log_volume_n_ball(ndims)
    logZ = gammaln(ndims/2) + log_volume_n_ball(ndims-1) - log_volume_n_ball(ndims, R)  + np.log(sigma)*ndims + (ndims/2-1)*np.log(2)
    plt.hist(samples.logZ(1000))
    samples = gaussian(nlive, ndims)
    samples
    samples.gui()
    gaussian(nlive, ndims).logZ()
    samples.gui()
    plt.plot(samples.logL, samples.nlive)
    samples.nlive.plot()
    samples.gui()


def test_perfect_ns_correlated_gaussian():
    np.random.seed(0)
    nlive = 500
    d = 5
    mean = 0.5*np.ones(d)
    cov = random_covariance(np.random.rand(d)*0.1)
    samples = correlated_gaussian(nlive, mean, cov)
    samples.nlive.plot()
    assert_allclose(samples.logZ(), 0, atol=3*samples.logZ(12).std())
    assert (samples[:-nlive].nlive >= nlive).all()
    assert_allclose(samples[[0, 1, 2, 3, 4]].mean(), mean, atol=1e-2)
    assert_allclose(samples[[0, 1, 2, 3, 4]].cov(), cov/(d+2), atol=1e-2)
