import pytest
import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from anesthetic.anesthetic import MCMCSamples, NestedSamples
from numpy.testing import assert_array_equal


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
