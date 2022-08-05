import numpy as np
from numpy.testing import assert_allclose
from anesthetic.examples.utils import (
    random_ellipsoid, random_covariance, volume_n_ball
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
    rvs = random_ellipsoid(mean, cov)

    assert(rvs(10).shape == (10, d))
    assert(rvs((10,)).shape == (10, d))
    assert(rvs((10, 20)).shape == (10, 20, d))

    dat = rvs(100000)

    atol = 1/np.sqrt(len(dat))
    assert_allclose(np.mean(dat, axis=0), mean, atol=atol)
    assert_allclose(np.cov(dat.T), cov/(d+2), atol=atol)

    cov = random_covariance(np.random.rand(2))
    rvs = random_ellipsoid([0, 0], cov)
    dat = rvs(1000000)
    hull = ConvexHull(dat)
    vol = hull.volume
    assert_allclose(vol, np.linalg.det(cov)**0.5*volume_n_ball(2), rtol=1e-3)
