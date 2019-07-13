import matplotlib_agg  # noqa: F401
import sys
import pytest
import numpy
import matplotlib.pyplot as plt
from anesthetic import MCMCSamples, NestedSamples, make_1d_axes, make_2d_axes
from numpy.testing import assert_array_equal
try:
    import montepython  # noqa: F401
except ImportError:
    pass


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
    plt.close("all")


@pytest.mark.xfail('montepython' not in sys.modules,
                   raises=ImportError,
                   reason="requires montepython package")
def test_read_montepython():
    mcmc = MCMCSamples(root='./tests/example_data/mp')
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")


def test_read_multinest():
    ns = NestedSamples(root='./tests/example_data/mn')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])

    ns = NestedSamples(root='./tests/example_data/mn_old')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")


def test_read_polychord():
    ns = NestedSamples(root='./tests/example_data/pc')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")


def test_different_parameters():
    params_x = ['x0', 'x1', 'x2', 'x3', 'x4']
    params_y = ['x0', 'x1', 'x2']
    fig, axes = make_1d_axes(params_x)
    ns = NestedSamples(root='./tests/example_data/pc')
    ns.plot_1d(axes)
    fig, axes = make_2d_axes(params_y)
    ns.plot_2d(axes)
    fig, axes = make_2d_axes(params_x)
    ns.plot_2d(axes)
    fig, axes = make_2d_axes([params_x, params_y])
    ns.plot_2d(axes)
    plt.close('all')


def test_plot_2d_types():
    ns = NestedSamples(root='./tests/example_data/pc')
    params_x = ['x0', 'x1', 'x2', 'x3']
    params_y = ['x0', 'x1', 'x2']
    params = [params_x, params_y]
    # Test old interface
    fig, axes = ns.plot_2d(params, types=['kde', 'scatter'])
    assert((~axes.isnull()).sum().sum() == 12)

    fig, axes = ns.plot_2d(params, types='kde')
    assert((~axes.isnull()).sum().sum() == 6)

    fig, axes = ns.plot_2d(params, types='kde', diagonal=False)
    assert((~axes.isnull()).sum().sum() == 3)

    # Test new interface
    fig, axes = ns.plot_2d(params, types={'lower': 'kde'})
    assert((~axes.isnull()).sum().sum() == 3)

    fig, axes = ns.plot_2d(params, types={'upper': 'scatter'})
    assert((~axes.isnull()).sum().sum() == 6)

    fig, axes = ns.plot_2d(params, types={'upper': 'kde', 'diagonal': 'kde'})
    assert((~axes.isnull()).sum().sum() == 9)

    fig, axes = ns.plot_2d(params, types={'lower': 'kde', 'diagonal': 'kde'})
    assert((~axes.isnull()).sum().sum() == 6)

    fig, axes = ns.plot_2d(params, types={'lower': 'kde', 'diagonal': 'kde',
                                          'upper': 'scatter'})
    assert((~axes.isnull()).sum().sum() == 12)
    plt.close("all")


def test_plot_2d_types_multiple_calls():
    ns = NestedSamples(root='./tests/example_data/pc')
    params = ['x0', 'x1', 'x2', 'x3']

    fig, axes = ns.plot_2d(params, types={'diagonal': 'kde',
                                          'lower': 'kde',
                                          'upper': 'scatter'})
    ns.plot_2d(axes, types={'diagonal': 'hist'})

    fig, axes = ns.plot_2d(params, types={'diagonal': 'hist'})
    ns.plot_2d(axes, types={'diagonal': 'kde',
                            'lower': 'kde',
                            'upper': 'scatter'})
    plt.close('all')
