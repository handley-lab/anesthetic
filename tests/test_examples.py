import numpy as np
from numpy.testing import assert_allclose
from anesthetic.examples.utils import (
    random_ellipsoid, random_covariance, volume_n_ball, log_volume_n_ball
)
from anesthetic.examples.perfect_ns import (
    gaussian, correlated_gaussian, wedding_cake, planck_gaussian
)
from scipy.spatial import ConvexHull
from scipy.special import logsumexp


def test_random_covariance():
    np.random.seed(0)
    d = 5
    sigmas = np.random.rand(d)
    cov = random_covariance(sigmas)
    assert cov.shape == (d, d)
    e = np.linalg.eigvals(cov)
    assert_allclose(np.sort(e), np.sort(sigmas**2))


def test_random_ellipsoid():
    np.random.seed(0)

    d = 5
    mean = np.random.rand(d)
    cov = random_covariance(np.random.rand(d))

    def rvs(shape=None):
        return random_ellipsoid(mean, cov, shape)

    assert rvs(10).shape == (10, d)
    assert rvs((10,)).shape == (10, d)
    assert rvs((10, 20)).shape == (10, 20, d)

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

    D_KL = log_volume_n_ball(ndims) - ndims/2*np.log(2*np.pi*np.e*sigma**2)
    assert_allclose(samples.D_KL(), D_KL, atol=3*samples.D_KL(12).std())

    assert_allclose(samples.d_G(), ndims, atol=3*samples.d_G(12).std())

    logZ = ndims/2 * np.log(2*np.pi*sigma**2) - log_volume_n_ball(ndims, R)
    assert_allclose(samples.logZ(), logZ, atol=3*samples.logZ(12).std())


def test_perfect_ns_correlated_gaussian():
    np.random.seed(0)
    nlive = 500
    ndims = 5
    mean = 0.5*np.ones(ndims)
    cov = random_covariance(np.random.rand(ndims)*0.1)
    samples = correlated_gaussian(nlive, mean, cov)
    samples.gui()
    logZ = np.linalg.slogdet(2*np.pi*cov)[1]/2
    assert_allclose(samples.logZ(), logZ, atol=3*samples.logZ(12).std())
    assert (samples[:-nlive].nlive >= nlive).all()
    assert_allclose(samples[np.arange(ndims)].mean(), mean, atol=1e-2)
    assert_allclose(samples[np.arange(ndims)].cov(), cov/(ndims+2), atol=1e-2)


def test_wedding_cake():
    np.random.seed(0)
    nlive = 500
    ndims = 4
    sigma = 0.01
    alpha = 0.5
    samples = wedding_cake(nlive, ndims, sigma, alpha)

    assert samples.nlive.iloc[0] == nlive
    assert samples.nlive.iloc[-1] == 1
    assert (samples.nlive <= nlive).all()

    mean = np.sqrt(ndims/2) * (np.log(4*ndims*sigma**2)-1) / np.log(alpha)
    std = -np.sqrt(ndims/2)/np.log(alpha)
    i = np.arange(mean + std*10)
    logZ = logsumexp(-alpha**(2*i/ndims)/8/sigma**2
                     + i*np.log(alpha) + np.log(1-alpha))

    assert_allclose(samples.logZ(), logZ, atol=3*samples.logZ(12).std())


def test_planck():
    np.random.seed(0)
    nlive = 50
    cov = np.array([
        [2.12e-08, -9.03e-08, 1.76e-08, 2.96e-07, 4.97e-07, 2.38e-07],
        [-9.03e-08, 1.39e-06, -1.26e-07, -3.41e-06, -4.15e-06, -3.28e-06],
        [1.76e-08, -1.26e-07, 9.71e-08, 4.30e-07, 7.41e-07, 4.13e-07],
        [2.96e-07, -3.41e-06, 4.30e-07, 5.33e-05, 9.53e-05, 1.05e-05],
        [4.97e-07, -4.15e-06, 7.41e-07, 9.53e-05, 2.00e-04, 1.35e-05],
        [2.38e-07, -3.28e-06, 4.13e-07, 1.05e-05, 1.35e-05, 1.73e-05]])

    mean = np.array([0.02237, 0.1200, 1.04092, 0.0544, 3.044, 0.9649])

    bounds = np.array([
        [5.00e-03, 1.00e-01],
        [1.00e-03, 9.90e-01],
        [5.00e-01, 1.00e+01],
        [1.00e-02, 8.00e-01],
        [1.61e+00, 3.91e+00],
        [8.00e-01, 1.20e+00]])

    logL_mean = -1400.35
    d = 6
    logdetroottwopicov = np.linalg.slogdet(2*np.pi*cov)[1]/2
    logZ = logL_mean + logdetroottwopicov + d/2
    logV = np.log(np.diff(bounds)).sum()
    D_KL = logV - logdetroottwopicov - d/2
    planck = planck_gaussian(nlive)

    params = ['omegabh2', 'omegach2', 'theta', 'tau', 'logA', 'ns']

    assert (planck[params] > bounds.T[0]).all().all()
    assert (planck[params] < bounds.T[1]).all().all()
    assert_allclose(planck[params].mean(), mean, atol=1e-3)
    assert_allclose(planck[params].cov(), cov, atol=1e-3)
    assert_allclose(planck.logL.mean(), logL_mean,
                    atol=3*planck.logL_P(12).std())
    assert_allclose(planck.logZ(), logZ, atol=3*planck.logZ(12).std())
    assert_allclose(planck.D_KL(), D_KL, atol=3*planck.D_KL(12).std())
    assert_allclose(planck.d_G(), len(params), atol=3*planck.d_G(12).std())
    assert_allclose(planck.logL_P(), logL_mean, atol=3*planck.logL_P(12).std())


def test_logLmax():
    nlive = 1000
    mean = [0.1, 0.3, 0.5]
    # cov = covariance matrix
    cov = np.array([[.01, 0.009, 0], [0.009, .01, 0], [0, 0, 0.1]])*0.01
    bounds = [[0, 1], [0, 1], [0, 1]]
    logLmax = 10
    # d = number of parameters
    d = len(mean)
    samples = correlated_gaussian(nlive, mean, cov, bounds=bounds,
                                  logLmax=logLmax)
    abs_err = samples.logL.std()/np.sqrt(samples.neff())
    atol = abs_err*3
    assert_allclose(samples.logL.mean(), logLmax-d/2, atol=atol)
