import matplotlib_agg  # noqa: F401
import os
import sys
import pytest
import numpy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from anesthetic import MCMCSamples, NestedSamples, make_1d_axes, make_2d_axes
from anesthetic.samples import merge_nested_samples
from numpy.testing import assert_array_equal, assert_array_almost_equal
from matplotlib.colors import to_hex
from scipy.stats import ks_2samp
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
    assert_array_equal(mcmc.columns, numpy.array([0, 1, 2], dtype=object))

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
    assert_array_equal(mcmc.columns, numpy.array([0, 1, 2, 'logL', 'weight'],
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

    ns = NestedSamples(data=samples, logL=logL, w=w)
    assert(len(ns) == nsamps)
    assert(numpy.all(numpy.isfinite(ns.logL)))
    logL[:10] = -1e300
    w[:10] = 0.
    mcmc = MCMCSamples(data=samples, logL=logL, w=w, logzero=-1e29)
    ns = NestedSamples(data=samples, logL=logL, w=w, logzero=-1e29)
    assert_array_equal(mcmc.columns, numpy.array([0, 1, 2, 'logL', 'weight'],
                                                 dtype=object))
    assert_array_equal(ns.columns, numpy.array([0, 1, 2, 'logL', 'weight'],
                                               dtype=object))
    assert(numpy.all(mcmc.logL[:10] == -numpy.inf))
    assert(numpy.all(ns.logL[:10] == -numpy.inf))
    assert(numpy.all(mcmc.logL[10:] == logL[10:]))
    assert(numpy.all(ns.logL[10:] == logL[10:]))

    mcmc = MCMCSamples(data=samples, logL=logL, w=w, logzero=-1e301)
    ns = NestedSamples(data=samples, logL=logL, w=w, logzero=-1e301)
    assert(numpy.all(numpy.isfinite(mcmc.logL)))
    assert(numpy.all(numpy.isfinite(ns.logL)))
    assert(numpy.all(mcmc.logL == logL))
    assert(numpy.all(ns.logL == logL))

    assert(mcmc.root is None)


def test_read_getdist():
    numpy.random.seed(3)
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
    numpy.random.seed(3)
    mcmc = MCMCSamples(root='./tests/example_data/mp')
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")


def test_read_multinest():
    numpy.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/mn')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])

    ns = NestedSamples(root='./tests/example_data/mn_old')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")


def test_read_polychord():
    numpy.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")

    os.rename('./tests/example_data/pc_phys_live-birth.txt',
              './tests/example_data/pc_phys_live-birth.txt_')
    ns_nolive = NestedSamples(root='./tests/example_data/pc')
    os.rename('./tests/example_data/pc_phys_live-birth.txt_',
              './tests/example_data/pc_phys_live-birth.txt')

    cols = ['x0', 'x1', 'x2', 'x3', 'x4', 'logL', 'logL_birth']
    assert_array_equal(ns_nolive[cols], ns[cols][:ns_nolive.shape[0]])


def test_NS_input_fails_in_MCMCSamples():
    with pytest.raises(ValueError) as excinfo:
        MCMCSamples(root='./tests/example_data/pc')
    assert "Please use NestedSamples instead which has the same features as " \
           "MCMCSamples and more. MCMCSamples should be used for MCMC " \
           "chains only." in str(excinfo.value)


def test_different_parameters():
    numpy.random.seed(3)
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


def test_manual_columns():
    old_params = ['x0', 'x1', 'x2', 'x3', 'x4']
    mcmc_params = ['logL', 'weight']
    ns_params = ['logL', 'logL_birth', 'nlive', 'weight']
    mcmc = MCMCSamples(root='./tests/example_data/gd')
    ns = NestedSamples(root='./tests/example_data/pc')
    assert_array_equal(mcmc.columns, old_params + mcmc_params)
    assert_array_equal(ns.columns, old_params + ns_params)

    new_params = ['y0', 'y1', 'y2', 'y3', 'y4']
    mcmc = MCMCSamples(root='./tests/example_data/gd', columns=new_params)
    ns = NestedSamples(root='./tests/example_data/pc', columns=new_params)
    assert_array_equal(mcmc.columns, new_params + mcmc_params)
    assert_array_equal(ns.columns, new_params + ns_params)


def test_plot_2d_types():
    numpy.random.seed(3)
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
    numpy.random.seed(3)
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


def test_root_and_label():
    numpy.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    assert(ns.root == './tests/example_data/pc')
    assert(ns.label == 'pc')

    ns = NestedSamples()
    assert(ns.root is None)
    assert(ns.label is None)

    mc = MCMCSamples(root='./tests/example_data/gd')
    assert (mc.root == './tests/example_data/gd')
    assert(mc.label == 'gd')

    mc = MCMCSamples()
    assert(mc.root is None)
    assert(mc.label is None)


def test_plot_2d_legend():
    numpy.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    mc = MCMCSamples(root='./tests/example_data/gd')
    params = ['x0', 'x1', 'x2', 'x3']

    # Test label kwarg for kde
    fig, axes = make_2d_axes(params, upper=False)
    ns.plot_2d(axes, label='l1', types=dict(diagonal='kde', lower='kde'))
    mc.plot_2d(axes, label='l2', types=dict(diagonal='kde', lower='kde'))

    for y, row in axes.iterrows():
        for x, ax in row.iteritems():
            if ax is not None:
                handles, labels = ax.get_legend_handles_labels()
                assert(labels == ['l1', 'l2'])
                if x == y:
                    assert(all([isinstance(h, Line2D) for h in handles]))
                else:
                    assert(all([isinstance(h, Rectangle) for h in handles]))
    plt.close('all')

    # Test label kwarg for hist and scatter
    fig, axes = make_2d_axes(params, lower=False)
    ns.plot_2d(axes, label='l1', types=dict(diagonal='hist', upper='scatter'))
    mc.plot_2d(axes, label='l2', types=dict(diagonal='hist', upper='scatter'))

    for y, row in axes.iterrows():
        for x, ax in row.iteritems():
            if ax is not None:
                handles, labels = ax.get_legend_handles_labels()
                assert(labels == ['l1', 'l2'])
                if x == y:
                    assert(all([isinstance(h, Rectangle) for h in handles]))
                else:
                    assert(all([isinstance(h, Line2D)
                                for h in handles]))
    plt.close('all')

    # test default labelling
    fig, axes = make_2d_axes(params, upper=False)
    ns.plot_2d(axes)
    mc.plot_2d(axes)

    for y, row in axes.iterrows():
        for x, ax in row.iteritems():
            if ax is not None:
                handles, labels = ax.get_legend_handles_labels()
                assert(labels == ['pc', 'gd'])
    plt.close('all')

    # Test label kwarg to constructors
    ns = NestedSamples(root='./tests/example_data/pc', label='l1')
    mc = MCMCSamples(root='./tests/example_data/gd', label='l2')
    params = ['x0', 'x1', 'x2', 'x3']

    fig, axes = make_2d_axes(params, upper=False)
    ns.plot_2d(axes)
    mc.plot_2d(axes)

    for y, row in axes.iterrows():
        for x, ax in row.iteritems():
            if ax is not None:
                handles, labels = ax.get_legend_handles_labels()
                assert(labels == ['l1', 'l2'])
    plt.close('all')


def test_plot_2d_colours():
    numpy.random.seed(3)
    gd = MCMCSamples(root="./tests/example_data/gd")
    gd.drop(columns='x3', inplace=True)
    pc = NestedSamples(root="./tests/example_data/pc")
    pc.drop(columns='x4', inplace=True)
    mn = NestedSamples(root="./tests/example_data/mn")
    mn.drop(columns='x2', inplace=True)

    plot_types = ['kde', 'hist']
    if 'fastkde' in sys.modules:
        plot_types += ['fastkde']

    for types in plot_types:
        fig = plt.figure()
        fig, axes = make_2d_axes(['x0', 'x1', 'x2', 'x3', 'x4'], fig=fig)
        types = {'diagonal': types, 'lower': types, 'upper': 'scatter'}
        gd.plot_2d(axes, types=types, label="gd")
        pc.plot_2d(axes, types=types, label="pc")
        mn.plot_2d(axes, types=types, label="mn")
        gd_colors = []
        pc_colors = []
        mn_colors = []
        for y, rows in axes.iterrows():
            for x, ax in rows.iteritems():
                handles, labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if isinstance(handle, Rectangle):
                        color = to_hex(handle.get_facecolor())
                    else:
                        color = handle.get_color()

                    if label == 'gd':
                        gd_colors.append(color)
                    elif label == 'pc':
                        pc_colors.append(color)
                    elif label == 'mn':
                        mn_colors.append(color)

        assert(len(set(gd_colors)) == 1)
        assert(len(set(mn_colors)) == 1)
        assert(len(set(pc_colors)) == 1)
        plt.close("all")


def test_plot_1d_colours():
    numpy.random.seed(3)
    gd = MCMCSamples(root="./tests/example_data/gd")
    gd.drop(columns='x3', inplace=True)
    pc = NestedSamples(root="./tests/example_data/pc")
    pc.drop(columns='x4', inplace=True)
    mn = NestedSamples(root="./tests/example_data/mn")
    mn.drop(columns='x2', inplace=True)

    plot_types = ['kde', 'hist']
    if 'astropy' in sys.modules:
        plot_types += ['astropyhist']
    if 'fastkde' in sys.modules:
        plot_types += ['fastkde']

    for plot_type in plot_types:
        fig = plt.figure()
        fig, axes = make_1d_axes(['x0', 'x1', 'x2', 'x3', 'x4'], fig=fig)
        gd.plot_1d(axes, plot_type=plot_type, label="gd")
        pc.plot_1d(axes, plot_type=plot_type, label="pc")
        mn.plot_1d(axes, plot_type=plot_type, label="mn")
        gd_colors = []
        pc_colors = []
        mn_colors = []
        for x, ax in axes.iteritems():
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if isinstance(handle, Rectangle):
                    color = to_hex(handle.get_facecolor())
                else:
                    color = handle.get_color()

                if label == 'gd':
                    gd_colors.append(color)
                elif label == 'pc':
                    pc_colors.append(color)
                elif label == 'mn':
                    mn_colors.append(color)

        assert(len(set(gd_colors)) == 1)
        assert(len(set(mn_colors)) == 1)
        assert(len(set(pc_colors)) == 1)
        plt.close("all")


@pytest.mark.xfail('astropy' not in sys.modules,
                   raises=ImportError,
                   reason="requires astropy package")
def test_astropyhist():
    numpy.random.seed(3)
    mcmc = NestedSamples(root='./tests/example_data/pc')
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'], types={'diagonal': 'astropyhist'})
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'], plot_type='astropyhist')
    plt.close("all")


def test_hist_levels():
    numpy.random.seed(3)
    mcmc = NestedSamples(root='./tests/example_data/pc')
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'], types={'lower': 'hist'},
                 levels=[0.68, 0.95], bins=20)
    plt.close("all")


def test_ns_output():
    numpy.random.seed(3)
    pc = NestedSamples(root='./tests/example_data/pc')
    for beta in [1., 0., 0.5]:
        pc.beta = beta
        n = 1000
        PC = pc.ns_output(n)
        assert abs(pc.logZ() - PC['logZ'].mean()) < PC['logZ'].std()
        assert PC['d'].mean() < 5
        assert PC.cov()['D']['logZ'] < 0
        assert(abs(PC.logZ.mean() - pc.logZ()) < PC.logZ.std() * n**0.5 * 2)
        assert(abs(PC.D.mean() - pc.D()) < PC.D.std() * n**0.5 * 2)
        assert(abs(PC.d.mean() - pc.d()) < PC.d.std() * n**0.5 * 2)

        n = 100
        assert(ks_2samp(pc.logZ(n), PC.logZ).pvalue > 0.05)
        assert(ks_2samp(pc.D(n), PC.D).pvalue > 0.05)
        assert(ks_2samp(pc.d(n), PC.d).pvalue > 0.05)

    assert abs(pc.set_beta(0.0).logZ()) < 1e-2
    assert pc.set_beta(0.9).logZ() < pc.set_beta(1.0).logZ()


def test_masking():
    pc = NestedSamples(root="./tests/example_data/pc")
    mask = pc['x0'] > 0

    plot_types = ['kde', 'hist']
    if 'fastkde' in sys.modules:
        plot_types += ['fastkde']

    for ptype in plot_types:
        fig, axes = make_1d_axes(['x0', 'x1', 'x2'])
        pc[mask].plot_1d(axes=axes, plot_type=ptype)

    for ptype in plot_types + ['scatter']:
        fig, axes = make_2d_axes(['x0', 'x1', 'x2'], upper=False)
        pc[mask].plot_2d(axes=axes, types=dict(lower=ptype, diagonal='hist'))


def test_merging():
    numpy.random.seed(3)
    samples_1 = NestedSamples(root='./tests/example_data/pc')
    samples_2 = NestedSamples(root='./tests/example_data/pc_250')
    samples = merge_nested_samples([samples_1, samples_2])
    nlive_1 = samples_1.nlive.mode()[0]
    nlive_2 = samples_2.nlive.mode()[0]
    nlive = samples.nlive.mode()[0]
    assert nlive_1 == 125
    assert nlive_2 == 250
    assert nlive == nlive_1 + nlive_2
    assert (samples.logZ() < samples_1.logZ()
            and samples.logZ() > samples_2.logZ()
            or samples.logZ() > samples_1.logZ()
            and samples.logZ() < samples_2.logZ())


def test_beta():
    pc = NestedSamples(root="./tests/example_data/pc")
    weight = pc.weight
    assert_array_equal(pc['weight'], pc.weight)
    assert pc.beta == 1

    prior = pc.set_beta(0)
    assert prior.beta == 0
    assert_array_equal(prior['weight'], prior.weight)
    assert pc.beta == 1
    assert_array_equal(pc['weight'], pc.weight)
    assert_array_almost_equal(sorted(prior.weight, reverse=True), prior.weight)

    for beta in numpy.linspace(0, 2, 10):
        pc.set_beta(beta, inplace=True)
        assert pc.beta == beta
        assert_array_equal(pc['weight'], pc.weight)
        assert not numpy.array_equal(pc['weight'], weight)

    for beta in numpy.linspace(0, 2, 10):
        pc.beta = beta
        assert pc.beta == beta
        assert_array_equal(pc['weight'], pc.weight)
        assert not numpy.array_equal(pc['weight'], weight)


def test_live_points():
    numpy.random.seed(4)
    pc = NestedSamples(root="./tests/example_data/pc")

    for i, logL in pc.logL.iteritems():
        live_points = pc.live_points(logL)
        assert len(live_points) == pc.nlive[i]

        live_points_from_int = pc.live_points(i)
        assert_array_equal(live_points_from_int, live_points)

    last_live_points = pc.live_points()
    logL = pc.logL_birth.max()
    assert (last_live_points.logL > logL).all()
    assert len(last_live_points) == pc.nlive.mode()[0]


def test_limit_assignment():
    numpy.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    # `None` in .ranges file:
    assert ns.limits['x0'][0] is None
    assert ns.limits['x0'][1] is None
    # parameter not listed in .ranges file:
    assert ns.limits['x1'][0] == ns.x1.min()
    assert ns.limits['x1'][1] == ns.x1.max()
    # `None` for only one limit in .ranges file:
    assert ns.limits['x2'][0] == 0
    assert ns.limits['x2'][1] is None
    # both limits specified in .ranges file:
    assert ns.limits['x3'][0] == 0
    assert ns.limits['x3'][1] == 1
