from anesthetic.examples.perfect_ns import correlated_gaussian
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from anesthetic.tension import tension_stats


def get_meanAB(meanA, meanB, covA, covB, covAB):
    meanAB = covAB@(np.linalg.solve(covA, meanA)+np.linalg.solve(covB, meanB))
    return meanAB


def get_covAB(covA, covB):
    covAB_inv = linalg.inv(covA) + linalg.inv(covB)
    covAB = linalg.inv(covAB_inv)
    return covAB


def get_logLmaxAB(meanA, meanB, covA, covB, logLmaxA=0, logLmaxB=0):
    meandiff = np.array(meanA)-np.array(meanB)
    logLmaxAB = -1/2 * meandiff@linalg.solve(covA+covB, meandiff)+logLmaxA
    +logLmaxB
    return logLmaxAB


def get_logR(meanA, meanB, covA, covB, V):
    meandiff = np.array(meanA)-np.array(meanB)
    logR = -1/2*meandiff@linalg.solve(covA+covB, meandiff)
    -1/2*np.linalg.slogdet(2*np.pi*(covA+covB))[1]+np.log(V)
    return logR


def get_logS(meanA, meanB, covA, covB, d):
    meandiff = np.array(meanA)-np.array(meanB)
    logS = d/2 - 1/2*meandiff@linalg.solve(covA+covB, meandiff)
    return logS


def get_logI(covA, covB, d, V):
    logI = - d/2 - 1/2 * np.log(linalg.det(2*np.pi*(covA + covB))) + np.log(V)
    return logI


def test_tension_stats_compatiable_gaussian(samples_plot=False,
                                             stats_table=False,
                                             stats_plot=False, hist_plot=False):
    nlive = 1000
    bounds = [[0, 1], [0, 1], [0, 1]]

    meanA = [0.1, 0.3, 0.5]
    covA = np.array([[.01, 0.009, 0], [0.009, .01, 0], [0, 0, 0.1]])*0.01
    logLmaxA = 0
    samplesA = correlated_gaussian(nlive, meanA, covA, bounds, logLmaxA)

    meanB = [0.1, 0.3, 0.5]
    covB = np.array([[.01, -0.009, 0.01], [-0.009, .01, -0.001], [0.01, -0.001, 0.1]])*0.01
    logLmaxB = 0
    samplesB = correlated_gaussian(nlive, meanB, covB, bounds, logLmaxB)
    
    covAB = get_covAB(covA, covB)
    meanAB = get_meanAB(meanA, meanB, covA, covB, covAB)
    logLmaxAB = get_logLmaxAB(meanA, meanB, covA, covB, logLmaxA=0, logLmaxB=0)
    samplesAB = correlated_gaussian(nlive, meanAB, covAB, bounds, logLmaxAB)

    nsamples = 1000
    beta = 1
    samples = tension_stats(samplesA, samplesB, samplesAB, nsamples, beta)

    logR_min = samples.logR.min()
    logR_max = samples.logR.max()

    logS_min = samples.logS.min()
    logS_max = samples.logS.max()

    logI_min = samples.logI.min()
    logI_max = samples.logI.max()

    logR_exact = get_logR(meanA, meanB, covA, covB, V=1)
    logS_exact = get_logS(meanA, meanB, covA, covB, d=len(bounds))
    logI_exact = get_logI(covA, covB, d=len(bounds), V=1)

    if samples_plot:
        axes = samplesA.plot_2d([0, 1, 2])
        samplesB.plot_2d(axes)
        samplesAB.plot_2d(axes)

    if stats_table:
        print(samples)

    if stats_plot:
        axes = samples.plot_2d(['logR', 'logI', 'logS', 'd_G'])

    if hist_plot:
        plt.figure()
        samples.logR.plot.hist()
        plt.axvline(logR_exact, color='r', label='Exact solution')
        plt.legend()
        plt.show()

        plt.figure()
        samples.logS.plot.hist()
        plt.axvline(logS_exact, color='r', label='Exact solution')
        plt.legend()
        plt.show()

        plt.figure()
        samples.logI.plot.hist()
        plt.axvline(logI_exact, color='r', label='Exact solution')
        plt.legend()
        plt.show()

    assert logR_min < logR_exact < logR_max
    assert logS_min < logS_exact < logS_max
    assert logI_min < logI_exact < logI_max


def test_tension_stats_incompatiable_gaussian(samples_plot=False,
                                               stats_table=False, stats_plot=False, hist_plot=False):
    nlive = 1000
    bounds = [[0, 1], [0, 1], [0, 1]]

    meanA = [0.1, 0.3, 0.5]
    covA = np.array([[.01, 0.009, 0], [0.009, .01, 0], [0, 0, 0.1]])*0.01
    logLmaxA = 0
    samplesA = correlated_gaussian(nlive, meanA, covA, bounds, logLmaxA)

    meanB = [0.15, 0.25, 0.45]
    covB = np.array([[.01, -0.009, 0.01], [-0.009, .01, -0.001], [0.01, -0.001, 0.1]])*0.01
    logLmaxB = 0
    samplesB = correlated_gaussian(nlive, meanB, covB, bounds, logLmaxB)

    covAB = get_covAB(covA, covB)
    meanAB = get_meanAB(meanA, meanB, covA, covB, covAB)
    logLmaxAB = get_logLmaxAB(meanA, meanB, covA, covB, logLmaxA=0, logLmaxB=0)
    samplesAB = correlated_gaussian(nlive, meanAB, covAB, bounds, logLmaxAB)

    nsamples = 1000
    beta = 1
    samples = tension_stats(samplesA, samplesB, samplesAB, nsamples, beta)

    logR_min = samples.logR.min()
    logR_max = samples.logR.max()

    logS_min = samples.logS.min()
    logS_max = samples.logS.max()

    logI_min = samples.logI.min()
    logI_max = samples.logI.max()

    logR_exact = get_logR(meanA, meanB, covA, covB, V=1)
    logS_exact = get_logS(meanA, meanB, covA, covB, d=len(bounds))
    logI_exact = get_logI(covA, covB, d=len(bounds), V=1)

    if samples_plot:
        axes = samplesA.plot_2d([0, 1, 2])
        samplesB.plot_2d(axes)
        samplesAB.plot_2d(axes)

    if stats_table:
        print(samples)

    if stats_plot:
        axes = samples.plot_2d(['logR', 'logI', 'logS', 'd_G'])

    if hist_plot:
        plt.figure()
        samples.logR.plot.hist()
        plt.axvline(logR_exact, color='r', label='Exact solution')
        plt.legend()
        plt.show()

        plt.figure()
        samples.logS.plot.hist()
        plt.axvline(logS_exact, color='r', label='Exact solution')
        plt.legend()
        plt.show()

        plt.figure()
        samples.logI.plot.hist()
        plt.axvline(logI_exact, color='r', label='Exact solution')
        plt.legend()
        plt.show()

    assert logR_min < logR_exact < logR_max
    assert logS_min < logS_exact < logS_max
    assert logI_min < logI_exact < logI_max
