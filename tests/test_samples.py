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

    mcmc = MCMCSamples(data=samples)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.columns, [0, 1, 2])

    mcmc = MCMCSamples(logL=logL)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.columns, ['logL'])

    mcmc = MCMCSamples(data=samples, logL=logL)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.columns, numpy.array([0, 1, 2, 'logL'],
                                                 dtype=object))

    mcmc = MCMCSamples(data=samples, w=w)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.columns, numpy.array([0, 1, 2, 'weight'],
                                                 dtype=object))

    mcmc = MCMCSamples(data=samples, w=w, logL=logL)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.columns, numpy.array([0, 1, 2, 'weight', 'logL'],
                                                 dtype=object))

    mcmc = MCMCSamples(data=samples, columns=params)
    assert(len(mcmc) == nsamps)
    assert_array_equal(mcmc.columns, ['A', 'B', 'C'])

    mcmc = MCMCSamples(data=samples, tex=tex)
    for p in params:
        assert(mcmc.tex[p] == tex[p])

    mcmc = MCMCSamples(data=samples, limits=limits)
    for p in params:
        assert(mcmc.limits[p] == limits[p])

    assert(mcmc.root is None)


def test_read_getdist():
    mcmc = MCMCSamples(root='./tests/example_data/gd')
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])

    mcmc = MCMCSamples(root='./tests/example_data/gd_single')
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])

def test_read_multinest():
    ns = NestedSamples(root='./tests/example_data/mn')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])

    ns = NestedSamples(root='./tests/example_data/mn_old')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])

def test_read_polychord():
    mcmc = NestedSamples(root='./tests/example_data/pc')
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])
