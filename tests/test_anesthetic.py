import pytest
import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from anesthetic.anesthetic import MCMCSamples, NestedSamples
from numpy.testing import assert_array_equal

def loglikelihood(x):
    sigma = 0.1
    return -(x-0.5) @ (x-0.5) / 2 / sigma**2

def ns_sim(ndims=4, nlive=50):
    """Brute force Nested Sampling run"""
    numpy.random.seed(0)
    live_points = numpy.random.rand(nlive, ndims)
    live_likes = numpy.array([loglikelihood(x) for x in live_points])
    live_birth_likes = numpy.ones(nlive) * -numpy.inf

    dead_points = []
    dead_likes = []
    birth_likes = []
    for _ in range(nlive*9):
        i = numpy.argmin(live_likes)
        Lmin = live_likes[i]
        dead_points.append(live_points[i].copy())
        dead_likes.append(live_likes[i])
        birth_likes.append(live_birth_likes[i])
        live_birth_likes[i] = Lmin
        while live_likes[i] <= Lmin:
            live_points[i, :] = numpy.random.rand(ndims)
            live_likes[i] = loglikelihood(live_points[i])
    return dead_points, dead_likes, birth_likes


def mcmc_sim(ndims=4):
    """ Simple Metropolis Hastings algorithm. """
    numpy.random.seed(0)
    x = [numpy.random.normal(0.5, 0.1, ndims)]
    l = [loglikelihood(x[-1])]
    w = [1]

    for _ in range(10000):
        x1 = x[-1] + numpy.random.randn(ndims)*0.1
        l1 = loglikelihood(x1)
        if numpy.random.rand() < numpy.exp(l1-l[-1]):
            x.append(x1)
            l.append(l1)
            w.append(1)
        else:
            w[-1]+=1
    return x, l, w


def test_build_mcmc():
    numpy.random.seed(3)
    nsamps = 1000
    ndims = 3
    params = numpy.random.randn(nsamps, ndims)
    logL = numpy.random.rand(nsamps)
    w = numpy.random.randint(1,20,size=nsamps)
    paramnames = ['A', 'B', 'C']
    tex = {'A':'$A$', 'B':'$B$', 'C':'$C$'}
    limits = {'A':(-1,1), 'B':(-2,2), 'C':(-3,3)}

    with pytest.raises(ValueError):
        mcmc = MCMCSamples.build()

    mcmc = MCMCSamples.build(params=params)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.paramnames, ['x0', 'x1', 'x2'])
    assert_array_equal(mcmc.columns, ['x0', 'x1', 'x2', 'u'])

    mcmc = MCMCSamples.build(logL=logL)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.paramnames, [])
    assert_array_equal(mcmc.columns, ['logL', 'u'])

    mcmc = MCMCSamples.build(params=params, logL=logL)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.paramnames, ['x0', 'x1', 'x2'])
    assert_array_equal(mcmc.columns, ['x0', 'x1', 'x2', 'logL', 'u'])

    mcmc = MCMCSamples.build(params=params, w=w)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.paramnames, ['x0', 'x1', 'x2'])
    assert_array_equal(mcmc.columns, ['x0', 'x1', 'x2', 'w', 'u'])

    mcmc = MCMCSamples.build(params=params, w=w, logL=logL)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.paramnames, ['x0', 'x1', 'x2'])
    assert_array_equal(mcmc.columns, ['x0', 'x1', 'x2', 'w', 'logL', 'u'])

    
    mcmc = MCMCSamples.build(params=params, paramnames=paramnames)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.paramnames, ['A', 'B', 'C'])
    assert_array_equal(mcmc.columns, ['A', 'B', 'C', 'u'])

    mcmc = MCMCSamples.build(params=params, tex=tex)
    for p in paramnames:
        assert(mcmc.tex[p] == tex[p])

    mcmc = MCMCSamples.build(params=params, limits=limits)
    for p in paramnames:
        assert(mcmc.limits[p] == limits[p])

    assert(mcmc.root is None)


def test_plot_mcmc():
    params, logL, w = mcmc_sim()
    mcmc = MCMCSamples.build(params=params, logL=logL, w=w)
    mcmc.plot_2d()
    mcmc.plot_1d()


def test_plot_ns():
    params, logL, logL_birth = ns_sim()
    ns = NestedSamples.build(params=params, logL=logL, logL_birth=logL_birth)
    ns.plot_2d()
    ns.plot_1d()
    ns.ns_output()
