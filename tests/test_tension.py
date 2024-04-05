from anesthetic.examples.perfect_ns import correlated_gaussian
import numpy as np
from numpy.linalg import inv, solve, slogdet
from anesthetic.tension import tension_stats


def test_tension_stats_compatiable_gaussian():
    d = 3
    V = 6.0
    nlive = 1000
    bounds = [[-1, 1], [0, 3], [0, 1]]

    meanA = [0.1, 0.3, 0.5]
    covA = np.array([[.01, 0.009, 0],
                     [0.009, .01, 0],
                     [0, 0, 0.1]])*0.01
    logLmaxA = 0
    samplesA = correlated_gaussian(nlive, meanA, covA, bounds, logLmaxA)

    meanB = [0.1, 0.3, 0.5]
    covB = np.array([[.01, -0.009, 0.01],
                     [-0.009, .01, -0.001],
                     [0.01, -0.001, 0.1]])*0.01
    logLmaxB = 0
    samplesB = correlated_gaussian(nlive, meanB, covB, bounds, logLmaxB)

    covAB = inv(inv(covA) + inv(covB))
    meanAB = covAB@(solve(covA, meanA)+solve(covB, meanB))
    dmeanAB = np.array(meanA)-np.array(meanB)
    logLmaxAB = -1/2 * dmeanAB@solve(covA+covB, dmeanAB) + logLmaxA + logLmaxB
    samplesAB = correlated_gaussian(nlive, meanAB, covAB, bounds, logLmaxAB)

    nsamples = 1000
    beta = 1
    samples_stats = tension_stats(samplesA, samplesB, samplesAB,
                                  nsamples, beta)

    logR_std = samples_stats.logR.std()
    logR_mean = samples_stats.logR.mean()
    logR_exact = (-1/2*dmeanAB@solve(covA+covB, dmeanAB) -
                  1/2*slogdet(2*np.pi*(covA+covB))[1] + np.log(V))
    assert logR_mean - 3 * logR_std < logR_exact < logR_mean + 3 * logR_std

    logS_std = samples_stats.logS.std()
    logS_mean = samples_stats.logS.mean()
    logS_exact = d/2 - 1/2*dmeanAB@solve(covA+covB, dmeanAB)
    assert logS_mean - 3 * logS_std < logS_exact < logS_mean + 3 * logS_std

    logI_std = samples_stats.logI.std()
    logI_mean = samples_stats.logI.mean()
    logI_exact = -d/2 - 1/2 * slogdet(2*np.pi*(covA+covB))[1] + np.log(V)
    assert logI_mean - 3 * logI_std < logI_exact < logI_mean + 3 * logI_std

    assert samples_stats.get_labels().tolist() == (['$\\log{R}$',
                                                    '$\\log{I}$', '$\\log{S}$',
                                                    '$d_G$', '$p$'])


def test_tension_stats_incompatiable_gaussian():
    np.random.rand(0)
    d = 3
    V = 6.0
    nlive = 1000
    bounds = [[-1, 1], [0, 3], [0, 1]]

    meanA = [0.1, 0.3, 0.5]
    covA = np.array([[.01, 0.009, 0],
                     [0.009, .01, 0],
                     [0, 0, 0.1]])*0.01
    logLmaxA = 0
    samplesA = correlated_gaussian(nlive, meanA, covA, bounds, logLmaxA)

    meanB = [0.15, 0.25, 0.45]
    covB = np.array([[.01, -0.009, 0.01],
                     [-0.009, .01, -0.001],
                     [0.01, -0.001, 0.1]])*0.01
    logLmaxB = 0
    samplesB = correlated_gaussian(nlive, meanB, covB, bounds, logLmaxB)

    covAB = inv(inv(covA) + inv(covB))
    meanAB = covAB@(solve(covA, meanA)+solve(covB, meanB))
    dmeanAB = np.array(meanA)-np.array(meanB)
    logLmaxAB = -1/2 * dmeanAB@solve(covA+covB, dmeanAB) + logLmaxA + logLmaxB
    samplesAB = correlated_gaussian(nlive, meanAB, covAB, bounds, logLmaxAB)

    nsamples = 1000
    beta = 1
    samples_stats = tension_stats(samplesA, samplesB, samplesAB,
                                  nsamples, beta)

    logR_std = samples_stats.logR.std()
    logR_mean = samples_stats.logR.mean()
    logR_exact = (-1/2*dmeanAB@solve(covA+covB, dmeanAB) -
                  1/2*slogdet(2*np.pi*(covA+covB))[1] + np.log(V))
    assert logR_mean - 3 * logR_std < logR_exact < logR_mean + 3 * logR_std

    logS_std = samples_stats.logS.std()
    logS_mean = samples_stats.logS.mean()
    logS_exact = d/2 - 1/2*dmeanAB@solve(covA+covB, dmeanAB)
    assert logS_mean - 3 * logS_std < logS_exact < logS_mean + 3 * logS_std

    logI_std = samples_stats.logI.std()
    logI_mean = samples_stats.logI.mean()
    logI_exact = -d/2 - 1/2 * slogdet(2*np.pi*(covA+covB))[1] + np.log(V)
    assert logI_mean - 3 * logI_std < logI_exact < logI_mean + 3 * logI_std

    assert samples_stats.get_labels().tolist() == ([r'$\log{R}$',
                                                    r'$\log{I}$', r'$\log{S}$',
                                                    r'$d_\mathrm{G}$', r'$p$'])
