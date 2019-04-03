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

