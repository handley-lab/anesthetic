#from anesthetic.samples import Samples
from anesthetic.examples.perfect_ns import correlated_gaussian
#from anesthetic import read_chains, make_2d_axes
import numpy as np
#import matplotlib.pyplot as plt
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
    logLmaxAB = (-1/2 * meandiff@linalg.solve(covA+covB, meandiff) +
                 logLmaxA + logLmaxB)
    return logLmaxAB


def get_logR(meanA, meanB, covA, covB, V):
    meandiff = np.array(meanA)-np.array(meanB)
    logR = (-1/2*meandiff@linalg.solve(covA+covB, meandiff) -
            1/2*np.linalg.slogdet(2*np.pi*(covA+covB))[1] + np.log(V))
    return logR

def get_logS(meanA, meanB, covA, covB, d):
    meandiff = np.array(meanA)-np.array(meanB)
    logS = d/2 - 1/2*meandiff@linalg.solve(covA+covB, meandiff)
    return logS

def get_logI(covA, covB, d, V):
    logI = -d/2 - 1/2 * np.log(linalg.det(2*np.pi*(covA+covB))) + np.log(V)
    return logI

def test_tension_stats_compatiable_gaussian():
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
   
    logR_std = samples.logR.std()
    logR_mean = samples.logR.mean()

    logS_std = samples.logS.std()
    logS_mean = samples.logS.mean()

    logI_std = samples.logI.std()
    logI_mean = samples.logI.mean()

    logR_exact = get_logR(meanA, meanB, covA, covB, V=1)
    logS_exact = get_logS(meanA, meanB, covA, covB, d=len(bounds))
    logI_exact = get_logI(covA, covB, d=len(bounds), V=1)

    assert logR_mean - 3 * logR_std < logR_exact < logR_mean + 3 * logR_std
    assert logS_mean - 3 * logS_std < logS_exact < logS_mean + 3 * logS_std
    assert logI_mean - 3 * logI_std < logI_exact < logI_mean + 3 * logI_std


def test_tension_stats_incompatiable_gaussian():
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

    logR_std = samples.logR.std()
    logR_mean = samples.logR.mean()

    logS_std = samples.logS.std()
    logS_mean = samples.logS.mean()

    logI_std = samples.logI.std()
    logI_mean = samples.logI.mean()

    logR_exact = get_logR(meanA, meanB, covA, covB, V=1)
    logS_exact = get_logS(meanA, meanB, covA, covB, d=len(bounds))
    logI_exact = get_logI(covA, covB, d=len(bounds), V=1)

    assert logR_mean - 3 * logR_std < logR_exact < logR_mean + 3 * logR_std
    assert logS_mean - 3 * logS_std < logS_exact < logS_mean + 3 * logS_std
    assert logI_mean - 3 * logI_std < logI_exact < logI_mean + 3 * logI_std
