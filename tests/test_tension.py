from pytest import approx
from anesthetic.examples.perfect_ns import correlated_gaussian
import numpy as np
from numpy.linalg import inv, slogdet
from anesthetic.tension import stats


def test_tension_stats_compatible_gaussian():
    np.random.seed(42)
    d = 3
    nlive = 10 * d
    bounds = [[-1, 1], [0, 3], [0, 1]]
    logV = np.log(np.diff(bounds).ravel().prod())

    muA = np.array([0.1, 0.3, 0.5])
    covA = 0.01 * np.array([[0.010, 0.009, 0.0],
                            [0.009, 0.010, 0.0],
                            [0.000, 0.000, 0.1]])
    invcovA = inv(covA)
    logLmaxA = 0
    samplesA = correlated_gaussian(nlive, muA, covA, bounds, logLmaxA)

    muB = np.array([0.1, 0.3, 0.5])
    covB = 0.01 * np.array([[+0.010, -0.009, +0.010],
                            [-0.009, +0.010, -0.001],
                            [+0.010, -0.001, +0.100]])
    invcovB = inv(covB)
    logLmaxB = 0
    samplesB = correlated_gaussian(nlive, muB, covB, bounds, logLmaxB)

    covAB = inv(invcovA + invcovB)
    muAB = covAB @ (invcovA @ muA + invcovB @ muB)
    dmuAB = muA - muB
    dmu_cov_dmu_AB = dmuAB @ invcovA @ covAB @ invcovB @ dmuAB
    logLmaxAB = logLmaxA + logLmaxB - dmu_cov_dmu_AB / 2
    samplesAB = correlated_gaussian(nlive, muAB, covAB, bounds, logLmaxAB)

    nsamples = 10
    beta = 1
    samples_stats = stats(samplesA, samplesB, samplesAB, nsamples, beta)

    logR_std = samples_stats.logR.std()
    logR_mean = samples_stats.logR.mean()
    logR_exact = logV - dmu_cov_dmu_AB/2 - slogdet(2*np.pi*(covA+covB))[1]/2
    assert logR_mean == approx(logR_exact, abs=3*logR_std)

    logS_std = samples_stats.logS.std()
    logS_mean = samples_stats.logS.mean()
    logS_exact = d / 2 - dmu_cov_dmu_AB / 2
    assert logS_mean == approx(logS_exact, abs=3*logS_std)

    logI_std = samples_stats.logI.std()
    logI_mean = samples_stats.logI.mean()
    logI_exact = logV - d / 2 - slogdet(2*np.pi*(covA+covB))[1] / 2
    assert logI_mean == approx(logI_exact, abs=3*logI_std)

    assert logS_mean == approx(logR_mean - logI_mean, abs=3*logS_std)

    assert samples_stats.get_labels().tolist() == ([r'$\ln\mathcal{R}$',
                                                    r'$\ln\mathcal{I}$',
                                                    r'$\ln\mathcal{S}$',
                                                    r'$d_\mathrm{G}$',
                                                    r'$p$'])


def test_tension_stats_incompatible_gaussian():
    np.random.seed(42)
    d = 3
    nlive = 10 * d
    bounds = [[-1, 1], [0, 3], [0, 1]]
    logV = np.log(np.diff(bounds).ravel().prod())

    muA = np.array([0.1, 0.3, 0.5])
    covA = 0.01 * np.array([[0.010, 0.009, 0.0],
                            [0.009, 0.010, 0.0],
                            [0.000, 0.000, 0.1]])
    invcovA = inv(covA)
    logLmaxA = 0
    samplesA = correlated_gaussian(nlive, muA, covA, bounds, logLmaxA)

    muB = np.array([0.15, 0.25, 0.45])
    covB = 0.01 * np.array([[+0.010, -0.009, +0.010],
                            [-0.009, +0.010, -0.001],
                            [+0.010, -0.001, +0.100]])
    invcovB = inv(covB)
    logLmaxB = 0
    samplesB = correlated_gaussian(nlive, muB, covB, bounds, logLmaxB)

    covAB = inv(invcovA + invcovB)
    muAB = covAB @ (invcovA @ muA + invcovB @ muB)
    dmuAB = muA - muB
    dmu_cov_dmu_AB = dmuAB @ invcovA @ covAB @ invcovB @ dmuAB
    logLmaxAB = logLmaxA + logLmaxB - dmu_cov_dmu_AB / 2
    samplesAB = correlated_gaussian(nlive, muAB, covAB, bounds, logLmaxAB)

    nsamples = 10
    beta = 1
    samples_stats = stats(samplesA, samplesB, samplesAB, nsamples, beta)

    logR_std = samples_stats.logR.std()
    logR_mean = samples_stats.logR.mean()
    logR_exact = logV - dmu_cov_dmu_AB/2 - slogdet(2*np.pi*(covA+covB))[1]/2
    assert logR_mean == approx(logR_exact, abs=3*logR_std)

    logS_std = samples_stats.logS.std()
    logS_mean = samples_stats.logS.mean()
    logS_exact = d / 2 - dmu_cov_dmu_AB / 2
    assert logS_mean == approx(logS_exact, abs=3*logS_std)

    logI_std = samples_stats.logI.std()
    logI_mean = samples_stats.logI.mean()
    logI_exact = logV - d / 2 - slogdet(2*np.pi*(covA+covB))[1] / 2
    assert logI_mean == approx(logI_exact, abs=3*logI_std)

    assert logS_mean == approx(logR_mean - logI_mean, abs=3*logS_std)

    assert samples_stats.get_labels().tolist() == ([r'$\ln\mathcal{R}$',
                                                    r'$\ln\mathcal{I}$',
                                                    r'$\ln\mathcal{S}$',
                                                    r'$d_\mathrm{G}$',
                                                    r'$p$'])
