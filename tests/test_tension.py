from pytest import approx, raises
from anesthetic.examples.perfect_ns import correlated_gaussian
import numpy as np
from numpy.linalg import inv, slogdet
from anesthetic.tension import tension_stats


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
    statsA = samplesA.stats(nsamples=nsamples)
    statsB = samplesB.stats(nsamples=nsamples)
    statsAB = samplesAB.stats(nsamples=nsamples)
    with raises(ValueError):
        tension_stats(samplesAB, statsA, statsB)
    with raises(ValueError):
        tension_stats(statsAB, samplesA, statsB)
    s = tension_stats(statsAB, statsA, statsB)

    logR_exact = logV - dmu_cov_dmu_AB/2 - slogdet(2*np.pi*(covA+covB))[1]/2
    assert s.logR.mean() == approx(logR_exact, abs=3*s.logR.std())

    logS_exact = d / 2 - dmu_cov_dmu_AB / 2
    assert s.logS.mean() == approx(logS_exact, abs=3*s.logS.std())

    I_exact = logV - d / 2 - slogdet(2*np.pi*(covA+covB))[1] / 2
    assert s.I.mean() == approx(I_exact, abs=3*s.I.std())

    assert s.logS.mean() == approx(s.logR.mean() - s.I.mean(),
                                   abs=3*s.logS.std())

    assert s.p.mean() > 0.05
    assert s.sigma.mean() < 1

    assert s.get_labels().tolist() == ([r'$\ln\mathcal{R}$',
                                        r'$\mathcal{I}$',
                                        r'$\ln\mathcal{S}$',
                                        r'$d_\mathrm{G}$',
                                        r'$p$',
                                        r'$\sigma$'])

    # Test for more than two datasets.
    tension_stats(statsAB, statsA, statsB, statsB)


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
    statsA = samplesA.stats(nsamples=nsamples)
    statsB = samplesB.stats(nsamples=nsamples)
    statsAB = samplesAB.stats(nsamples=nsamples)
    s = tension_stats(statsAB, statsA, statsB)

    logR_exact = logV - dmu_cov_dmu_AB/2 - slogdet(2*np.pi*(covA+covB))[1]/2
    assert s.logR.mean() == approx(logR_exact, abs=3*s.logR.std())

    logS_exact = d / 2 - dmu_cov_dmu_AB / 2
    assert s.logS.mean() == approx(logS_exact, abs=3*s.logS.std())

    I_exact = logV - d / 2 - slogdet(2*np.pi*(covA+covB))[1] / 2
    assert s.I.mean() == approx(I_exact, abs=3*s.I.std())

    assert s.logS.mean() == approx(s.logR.mean() - s.I.mean(),
                                   abs=3*s.logS.std())

    assert s.p.mean() < 0.003
    assert s.sigma.mean() > 3

    assert s.get_labels().tolist() == ([r'$\ln\mathcal{R}$',
                                        r'$\mathcal{I}$',
                                        r'$\ln\mathcal{S}$',
                                        r'$d_\mathrm{G}$',
                                        r'$p$',
                                        r'$\sigma$'])
