from anesthetic.examples.perfect_ns import correlated_gaussian
import numpy as np
from numpy.testing import assert_allclose


def test_logLmax(print_result=False):
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
    if print_result:
        print(samples.logL.mean())
        print(logLmax-d/2)
    assert_allclose(samples.logL.mean(), logLmax-d/2, atol=atol)
