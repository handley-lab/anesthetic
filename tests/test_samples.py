import pytest
import numpy
from anesthetic import MCMCSamples, NestedSamples
from numpy.testing import assert_array_equal


def test_build_mcmc():
    numpy.random.seed(3)
    nsamps = 1000
    ndims = 3
    samples = numpy.random.randn(nsamps, ndims)
    logL = numpy.random.rand(nsamps)
    w = numpy.random.randint(1, 20, size=nsamps)
    params = ['A', 'B', 'C']
    tex = {'A': '$A$', 'B': '$B$', 'C': '$C$'}
    limits = {'A': (-1, 1), 'B': (-2, 2), 'C': (-3, 3)}

    with pytest.raises(ValueError):
        mcmc = MCMCSamples.build()

    mcmc = MCMCSamples.build(samples=samples)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.columns, ['x0', 'x1', 'x2'])

    mcmc = MCMCSamples.build(logL=logL)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.columns, ['logL'])

    mcmc = MCMCSamples.build(samples=samples, logL=logL)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.columns, ['x0', 'x1', 'x2', 'logL'])

    mcmc = MCMCSamples.build(samples=samples, w=w)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.columns, ['x0', 'x1', 'x2', 'weight'])

    mcmc = MCMCSamples.build(samples=samples, w=w, logL=logL)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.columns, ['x0', 'x1', 'x2', 'weight', 'logL'])

    mcmc = MCMCSamples.build(samples=samples, params=params)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.columns, ['A', 'B', 'C'])

    mcmc = MCMCSamples.build(samples=samples, tex=tex)
    for p in params:
        assert(mcmc.tex[p] == tex[p])

    mcmc = MCMCSamples.build(samples=samples, limits=limits)
    for p in params:
        assert(mcmc.limits[p] == limits[p])

    assert(mcmc.root is None)


def test_read_mcmc():
    mcmc = MCMCSamples.read('./tests/example_data/mcmc/mcmc')
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])


def test_plot_ns():
    ns = NestedSamples.read('./tests/example_data/ns/ns')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])
    ns.ns_output()
