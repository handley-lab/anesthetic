import warnings
import matplotlib_agg  # noqa: F401
import sys
import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from anesthetic import Samples, MCMCSamples, NestedSamples
from anesthetic import make_1d_axes, make_2d_axes
from anesthetic.samples import merge_nested_samples
from anesthetic.samples import merge_samples_weighted
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_array_less)
from pandas.testing import assert_frame_equal
from matplotlib.colors import to_hex
from scipy.stats import ks_2samp, kstest, norm
from wedding_cake import WeddingCake
try:
    import montepython  # noqa: F401
except ImportError:
    pass


def test_build_samples():
    np.random.seed(3)
    nsamps = 1000
    ndims = 3
    data = np.random.randn(nsamps, ndims)
    logL = np.random.rand(nsamps)
    weights = np.random.randint(1, 20, size=nsamps)
    params = ['A', 'B', 'C']
    tex = {'A': '$A$', 'B': '$B$', 'C': '$C$'}

    s = Samples(data=data)
    assert len(s) == nsamps
    assert_array_equal(s.columns, np.array([0, 1, 2], dtype=object))

    s = Samples(data=data, logL=logL)
    assert len(s) == nsamps
    assert_array_equal(s.columns, np.array([0, 1, 2, 'logL'], dtype=object))

    s = Samples(data=data, weights=weights)
    assert len(s) == nsamps
    assert_array_equal(s.columns, np.array([0, 1, 2], dtype=object))
    assert s.index.nlevels == 2

    s = Samples(data=data, weights=weights, logL=logL)
    assert len(s) == nsamps
    assert_array_equal(s.columns, np.array([0, 1, 2, 'logL'], dtype=object))
    assert s.index.nlevels == 2

    s = Samples(data=data, columns=params)
    assert len(s) == nsamps
    assert_array_equal(s.columns, ['A', 'B', 'C'])

    s = Samples(data=data, tex=tex)
    for p in params:
        assert s.tex[p] == tex[p]

    mc = MCMCSamples(data=data, logL=logL, weights=weights)
    assert len(mc) == nsamps
    assert np.all(np.isfinite(mc.logL))

    ns = NestedSamples(data=data, logL=logL, weights=weights)
    assert len(ns) == nsamps
    assert np.all(np.isfinite(ns.logL))

    logL[:10] = -1e300
    weights[:10] = 0.
    mc = MCMCSamples(data=data, logL=logL, weights=weights, logzero=-1e29)
    ns = NestedSamples(data=data, logL=logL, weights=weights, logzero=-1e29)
    assert_array_equal(mc.columns, np.array([0, 1, 2, 'logL'], dtype=object))
    assert_array_equal(ns.columns, np.array([0, 1, 2, 'logL'], dtype=object))
    assert mc.index.nlevels == 2
    assert ns.index.nlevels == 2
    assert np.all(mc.logL[:10] == -np.inf)
    assert np.all(ns.logL[:10] == -np.inf)
    assert np.all(mc.logL[10:] == logL[10:])
    assert np.all(ns.logL[10:] == logL[10:])

    mc = MCMCSamples(data=data, logL=logL, weights=weights, logzero=-1e301)
    ns = NestedSamples(data=data, logL=logL, weights=weights, logzero=-1e301)
    assert np.all(np.isfinite(mc.logL))
    assert np.all(np.isfinite(ns.logL))
    assert np.all(mc.logL == logL)
    assert np.all(ns.logL == logL)
    assert mc.root is None
    assert ns.root is None


def test_NS_input_fails_in_MCMCSamples():
    with pytest.raises(ValueError) as excinfo:
        MCMCSamples(root='./tests/example_data/pc')
    assert "Please use NestedSamples instead which has the same features as " \
           "Samples and more. MCMCSamples should be used for MCMC " \
           "chains only." in str(excinfo.value)


def test_different_parameters():
    np.random.seed(3)
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
    mcmc_params = ['logL']
    ns_params = ['logL', 'logL_birth', 'nlive']
    mcmc = MCMCSamples(root='./tests/example_data/gd')
    ns = NestedSamples(root='./tests/example_data/pc')
    assert_array_equal(mcmc.columns, old_params + mcmc_params)
    assert_array_equal(ns.columns, old_params + ns_params)

    new_params = ['y0', 'y1', 'y2', 'y3', 'y4']
    mcmc = MCMCSamples(root='./tests/example_data/gd', columns=new_params)
    ns = NestedSamples(root='./tests/example_data/pc', columns=new_params)
    assert_array_equal(mcmc.columns, new_params + mcmc_params)
    assert_array_equal(ns.columns, new_params + ns_params)


def test_plot_2d_kinds():
    np.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    params_x = ['x0', 'x1', 'x2', 'x3']
    params_y = ['x0', 'x1', 'x2']
    params = [params_x, params_y]

    # Check dictionaries
    fig, axes = ns.plot_2d(params, kind={'lower': 'kde_2d'})
    assert (~axes.isnull()).to_numpy().sum() == 3

    fig, axes = ns.plot_2d(params, kind={'upper': 'scatter_2d'})
    assert (~axes.isnull()).to_numpy().sum() == 6

    fig, axes = ns.plot_2d(params, kind={'upper': 'kde_2d',
                                         'diagonal': 'kde_1d'})
    assert (~axes.isnull()).to_numpy().sum() == 9

    fig, axes = ns.plot_2d(params, kind={'lower': 'kde_2d',
                                         'diagonal': 'kde_1d'})
    assert (~axes.isnull()).to_numpy().sum() == 6

    fig, axes = ns.plot_2d(params, kind={'lower': 'kde_2d',
                                         'diagonal': 'kde_1d'})
    assert (~axes.isnull()).to_numpy().sum() == 6

    fig, axes = ns.plot_2d(params, kind={'lower': 'kde_2d',
                                         'diagonal': 'kde_1d',
                                         'upper': 'scatter_2d'})
    assert (~axes.isnull()).to_numpy().sum() == 12

    # Check strings
    fig, axes = ns.plot_2d(params, kind='kde')
    assert (~axes.isnull()).to_numpy().sum() == 6
    fig, axes = ns.plot_2d(params, kind='kde_1d')
    assert (~axes.isnull()).to_numpy().sum() == 3
    fig, axes = ns.plot_2d(params, kind='kde_2d')
    assert (~axes.isnull()).to_numpy().sum() == 3

    # Check kinds vs kind kwarg
    fig, axes = ns.plot_2d(params, kinds='kde')
    assert (~axes.isnull()).to_numpy().sum() == 6
    fig, axes = ns.plot_2d(params, kinds='kde_1d')
    assert (~axes.isnull()).to_numpy().sum() == 3
    fig, axes = ns.plot_2d(params, kinds='kde_2d')
    assert (~axes.isnull()).to_numpy().sum() == 3

    # Check incorrect inputs
    with pytest.raises(ValueError):
        ns.plot_2d(params, kind={'lower': 'not a plot kind'})
    with pytest.raises(ValueError):
        ns.plot_2d(params, kind={'diagonal': 'not a plot kind'})
    with pytest.raises(ValueError):
        ns.plot_2d(params, kind={'lower': 'kde', 'spam': 'kde'})
    with pytest.raises(ValueError):
        ns.plot_2d(params, kind={'ham': 'kde'})
    with pytest.raises(ValueError):
        ns.plot_2d(params, kind=0)
    with pytest.raises(ValueError):
        ns.plot_2d(params, kind='eggs')

    plt.close("all")


def test_plot_2d_kinds_multiple_calls():
    np.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    params = ['x0', 'x1', 'x2', 'x3']

    fig, axes = ns.plot_2d(params, kind={'diagonal': 'kde_1d',
                                         'lower': 'kde_2d',
                                         'upper': 'scatter_2d'})
    ns.plot_2d(axes, kind={'diagonal': 'hist'})

    fig, axes = ns.plot_2d(params, kind={'diagonal': 'hist'})
    ns.plot_2d(axes, kind={'diagonal': 'kde_1d',
                           'lower': 'kde_2d',
                           'upper': 'scatter_2d'})
    plt.close('all')


def test_root_and_label():
    np.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    assert ns.root == './tests/example_data/pc'
    assert ns.label == 'pc'

    ns = NestedSamples()
    assert ns.root is None
    assert ns.label is None

    mc = MCMCSamples(root='./tests/example_data/gd')
    assert (mc.root == './tests/example_data/gd')
    assert mc.label == 'gd'

    mc = MCMCSamples()
    assert mc.root is None
    assert mc.label is None


def test_plot_2d_legend():
    np.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    mc = MCMCSamples(root='./tests/example_data/gd')
    params = ['x0', 'x1', 'x2', 'x3']

    # Test label kwarg for kde
    fig, axes = make_2d_axes(params, upper=False)
    ns.plot_2d(axes, label='l1', kind=dict(diagonal='kde_1d', lower='kde_2d'))
    mc.plot_2d(axes, label='l2', kind=dict(diagonal='kde_1d', lower='kde_2d'))

    for y, row in axes.iterrows():
        for x, ax in row.iteritems():
            if ax is not None:
                leg = ax.legend()
                assert leg.get_texts()[0].get_text() == 'l1'
                assert leg.get_texts()[1].get_text() == 'l2'
                handles, labels = ax.get_legend_handles_labels()
                assert labels == ['l1', 'l2']
                if x == y:
                    assert all([isinstance(h, Line2D) for h in handles])
                else:
                    assert all([isinstance(h, Rectangle) for h in handles])

    plt.close('all')

    # Test label kwarg for hist and scatter
    fig, axes = make_2d_axes(params, lower=False)
    ns.plot_2d(axes, label='l1', kind=dict(diagonal='hist_1d',
                                           upper='scatter_2d'))
    mc.plot_2d(axes, label='l2', kind=dict(diagonal='hist_1d',
                                           upper='scatter_2d'))

    for y, row in axes.iterrows():
        for x, ax in row.iteritems():
            if ax is not None:
                leg = ax.legend()
                assert leg.get_texts()[0].get_text() == 'l1'
                assert leg.get_texts()[1].get_text() == 'l2'
                handles, labels = ax.get_legend_handles_labels()
                assert labels == ['l1', 'l2']
                if x == y:
                    assert all([isinstance(h, Rectangle) for h in handles])
                else:
                    assert all([isinstance(h, Line2D)
                                for h in handles])
    plt.close('all')

    # test default labelling
    fig, axes = make_2d_axes(params, upper=False)
    ns.plot_2d(axes)
    mc.plot_2d(axes)

    for y, row in axes.iterrows():
        for x, ax in row.iteritems():
            if ax is not None:
                handles, labels = ax.get_legend_handles_labels()
                assert labels == ['pc', 'gd']
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
                assert labels == ['l1', 'l2']
    plt.close('all')


def test_plot_2d_colours():
    np.random.seed(3)
    gd = MCMCSamples(root="./tests/example_data/gd")
    gd.drop(columns='x3', inplace=True)
    pc = NestedSamples(root="./tests/example_data/pc")
    pc.drop(columns='x4', inplace=True)
    mn = NestedSamples(root="./tests/example_data/mn")
    mn.drop(columns='x2', inplace=True)

    kinds = ['kde', 'hist']
    if 'fastkde' in sys.modules:
        kinds += ['fastkde']

    for kind in kinds:
        fig = plt.figure()
        fig, axes = make_2d_axes(['x0', 'x1', 'x2', 'x3', 'x4'], fig=fig)
        kind = {'diagonal': kind + '_1d',
                'lower': kind + '_2d',
                'upper': 'scatter_2d'}
        gd.plot_2d(axes, kind=kind, label="gd")
        pc.plot_2d(axes, kind=kind, label="pc")
        mn.plot_2d(axes, kind=kind, label="mn")
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

        assert len(set(gd_colors)) == 1
        assert len(set(mn_colors)) == 1
        assert len(set(pc_colors)) == 1
        plt.close("all")


def test_plot_1d_colours():
    np.random.seed(3)
    gd = MCMCSamples(root="./tests/example_data/gd")
    gd.drop(columns='x3', inplace=True)
    pc = NestedSamples(root="./tests/example_data/pc")
    pc.drop(columns='x4', inplace=True)
    mn = NestedSamples(root="./tests/example_data/mn")
    mn.drop(columns='x2', inplace=True)

    kinds = ['kde', 'hist']
    if 'fastkde' in sys.modules:
        kinds += ['fastkde']

    for kind in kinds:
        fig = plt.figure()
        fig, axes = make_1d_axes(['x0', 'x1', 'x2', 'x3', 'x4'], fig=fig)
        gd.plot_1d(axes, kind=kind + '_1d', label="gd")
        pc.plot_1d(axes, kind=kind + '_1d', label="pc")
        mn.plot_1d(axes, kind=kind + '_1d', label="mn")
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

        assert len(set(gd_colors)) == 1
        assert len(set(mn_colors)) == 1
        assert len(set(pc_colors)) == 1
        plt.close("all")


@pytest.mark.xfail('astropy' not in sys.modules,
                   raises=ImportError,
                   reason="requires astropy package")
def test_astropyhist():
    np.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'], kind='hist_1d', bins='knuth')
    plt.close("all")


def test_hist_levels():
    np.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'], kind={'lower': 'hist_2d'},
               levels=[0.95, 0.68], bins=20)
    plt.close("all")


def test_ns_output():
    np.random.seed(3)
    pc = NestedSamples(root='./tests/example_data/pc')
    for beta in [1., 0., 0.5]:
        pc.beta = beta
        n = 1000
        PC = pc.ns_output(n)
        assert abs(pc.logZ() - PC['logZ'].mean()) < PC['logZ'].std()
        assert PC['d'].mean() < 5
        assert PC.cov()['D']['logZ'] < 0
        assert abs(PC.logZ.mean() - pc.logZ()) < PC.logZ.std() * n**0.5 * 2
        assert abs(PC.D.mean() - pc.D()) < PC.D.std() * n**0.5 * 2
        assert abs(PC.d.mean() - pc.d()) < PC.d.std() * n**0.5 * 2

        n = 100
        assert ks_2samp(pc.logZ(n), PC.logZ).pvalue > 0.05
        assert ks_2samp(pc.D(n), PC.D).pvalue > 0.05
        assert ks_2samp(pc.d(n), PC.d).pvalue > 0.05

    assert abs(pc.set_beta(0.0).logZ()) < 1e-2
    assert pc.set_beta(0.9).logZ() < pc.set_beta(1.0).logZ()

    assert_array_almost_equal(pc.set_beta(1).weights, pc.set_beta(1).weights)
    assert_array_almost_equal(pc.set_beta(.5).weights, pc.set_beta(.5).weights)
    assert_array_equal(pc.set_beta(0).weights, pc.set_beta(0).weights)


def test_masking():
    pc = NestedSamples(root="./tests/example_data/pc")
    mask = pc['x0'] > 0

    kinds = ['kde', 'hist']
    if 'fastkde' in sys.modules:
        kinds += ['fastkde']

    for kind in kinds:
        fig, axes = make_1d_axes(['x0', 'x1', 'x2'])
        pc[mask].plot_1d(axes=axes, kind=kind + '_1d')

    for kind in kinds + ['scatter']:
        fig, axes = make_2d_axes(['x0', 'x1', 'x2'], upper=False)
        pc[mask].plot_2d(axes=axes, kind=dict(lower=kind + '_2d',
                                              diagonal='hist_1d'))


def test_merging():
    np.random.seed(3)
    samples_1 = NestedSamples(root='./tests/example_data/pc')
    samples_2 = NestedSamples(root='./tests/example_data/pc_250')
    samples = merge_nested_samples([samples_1, samples_2])
    nlive_1 = samples_1.nlive.mode()[0]
    nlive_2 = samples_2.nlive.mode()[0]
    nlive = samples.nlive.mode()[0]
    assert nlive_1 == 125
    assert nlive_2 == 250
    assert nlive == nlive_1 + nlive_2
    assert (samples_1.logZ() > samples.logZ() > samples_2.logZ()
            or samples_1.logZ() < samples.logZ() < samples_2.logZ())
    assert 'x0' in samples.tex


def test_weighted_merging():
    # Generate some data to try it out:
    samples_1 = NestedSamples(root='./tests/example_data/pc')
    samples_2 = NestedSamples(root='./tests/example_data/pc_250')
    samples_1['xtest'] = 7*samples_1['x3']
    samples_2['xtest'] = samples_2['x3']
    samples_1.tex['xtest'] = "$x_{t,1}$"
    samples_2.tex['xtest'] = "$x_{t,2}$"
    mean1 = samples_1.mean()['xtest']
    mean2 = samples_2.mean()['xtest']

    # Test with evidence weights
    weight1 = np.exp(samples_1.logZ())
    weight2 = np.exp(samples_2.logZ())
    samples = merge_samples_weighted([samples_1, samples_2],
                                     label='Merged label')
    mean = samples.mean()['xtest']
    assert np.isclose(mean, (mean1*weight1+mean2*weight2)/(weight1+weight2))

    # Test tex and label
    for key in samples.keys():
        if key in samples_2.keys():
            assert samples.tex[key] == samples_2.tex[key]
        else:
            assert samples.tex[key] == samples_1.tex[key]
    assert samples.label == 'Merged label'

    # Test that label is None when no label is passed
    samples_1.label = "1"
    samples_2.label = "2"
    samples = merge_samples_weighted([samples_1, samples_2])
    assert samples.label is None

    # Test with explicit weights
    weight1 = 31
    weight2 = 13
    samples = merge_samples_weighted(
        [samples_1, samples_2], weights=[weight1, weight2])
    mean = samples.mean()['xtest']
    assert np.isclose(mean, (mean1*weight1+mean2*weight2)/(weight1+weight2))

    # Test plot still works (see issue #189)
    prior_samples = []
    for i in range(3):
        d = {"x": np.random.uniform(size=1000),
             "y": np.random.uniform(size=1000)}
        tmp = Samples(d)
        prior_samples.append(tmp)
    merge_prior = merge_samples_weighted(prior_samples, weights=np.ones(3))
    merge_prior.plot_2d(["x", "y"])
    plt.close('all')

    # Test if correct exceptions are raised:
    # MCMCSamples are passed without weights
    with pytest.raises(ValueError):
        merge_samples_weighted([MCMCSamples(samples_1)])
    # len(weights) != len(samples)
    with pytest.raises(ValueError):
        merge_samples_weighted([samples_1, samples_2], weights=[1, 2, 3])
    # A samples is passed and not a sequence
    with pytest.raises(TypeError):
        merge_samples_weighted(samples_1, weights=[1, 2, 3])


def test_beta():
    pc = NestedSamples(root="./tests/example_data/pc")
    weights = pc.weights
    assert_array_equal(weights, pc.weights)
    assert_array_equal(pc.index.get_level_values('weights'), pc.weights)
    assert pc.beta == 1

    prior = pc.set_beta(0)
    assert prior.beta == 0
    assert_array_equal(prior.index.get_level_values('weights'), prior.weights)
    assert pc.beta == 1
    assert_array_equal(pc.index.get_level_values('weights'), pc.weights)
    assert_array_almost_equal(sorted(prior.weights, reverse=True),
                              prior.weights)

    for beta in np.linspace(0, 2, 10):
        pc.set_beta(beta, inplace=True)
        assert pc.beta == beta
        assert_array_equal(pc.index.get_level_values('weights'), pc.weights)
        assert not np.array_equal(pc.index.get_level_values('weights'),
                                  weights)

    for beta in np.linspace(0, 2, 10):
        pc.beta = beta
        assert pc.beta == beta
        assert_array_equal(pc.index.get_level_values('weights'), pc.weights)
        assert not np.array_equal(pc.index.get_level_values('weights'),
                                  weights)


def test_beta_with_logL_infinities():
    ns = NestedSamples(root="./tests/example_data/pc")
    for i in range(10):
        ns.loc[i, 'logL'] = -np.inf
    prior = ns.set_beta(0)
    assert np.all(prior.logL[:10] == -np.inf)
    assert np.all(prior.weights[:10] == 0)
    ns.plot_1d(['x0', 'x1'])


def test_prior():
    ns = NestedSamples(root="./tests/example_data/pc")
    prior = ns.prior()
    assert prior.beta == 0
    assert_frame_equal(prior, ns.set_beta(0))


def test_live_points():
    np.random.seed(4)
    pc = NestedSamples(root="./tests/example_data/pc")

    for i, logL in pc.logL.iloc[::49].iteritems():
        live_points = pc.live_points(logL)
        assert len(live_points) == int(pc.nlive[i[0]])

        live_points_from_int = pc.live_points(i[0])
        assert_array_equal(live_points_from_int, live_points)

        live_points_from_index = pc.live_points(i)
        assert_array_equal(live_points_from_index, live_points)

    assert pc.live_points(0).index[0][0] == 0

    last_live_points = pc.live_points()
    logL = pc.logL_birth.max()
    assert (last_live_points.logL >= logL).all()
    assert len(last_live_points) == pc.nlive.mode()[0]


def test_hist_range_1d():
    """Test to provide a solution to #89"""
    np.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    fig, ax = ns.plot_1d('x0', kind='hist_1d')
    x1, x2 = ax['x0'].get_xlim()
    assert x1 > -1
    assert x2 < +1
    fig, ax = ns.plot_1d('x0', kind='hist_1d', bins=np.linspace(-1, 1, 11))
    x1, x2 = ax['x0'].get_xlim()
    assert x1 <= -1
    assert x2 >= +1


def test_contour_plot_2d_nan():
    """Contour plots with nans arising from issue #96"""
    np.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')

    ns.loc[:9, 'x0'] = np.nan
    with pytest.raises((np.linalg.LinAlgError, RuntimeError, ValueError)):
        ns.plot_2d(['x0', 'x1'])

    # Check this error is removed in the case of zero weights
    weights = ns.weights
    weights[:10] = 0
    ns.weights = weights
    ns.plot_2d(['x0', 'x1'])


def test_compute_insertion():
    np.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    assert 'insertion' not in ns
    ns._compute_insertion_indexes()
    assert 'insertion' in ns

    nlive = ns.nlive.mode()[0]
    assert_array_less(ns.insertion, nlive)

    u = ns.insertion.to_numpy()/nlive
    assert kstest(u[nlive:-nlive], 'uniform').pvalue > 0.05

    pvalues = [kstest(u[i:i+nlive], 'uniform').pvalue
               for i in range(nlive, len(ns)-2*nlive, nlive)]

    assert kstest(pvalues, 'uniform').pvalue > 0.05


def test_posterior_points():
    np.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    assert_array_equal(ns.posterior_points(), ns.posterior_points())
    assert_array_equal(ns.posterior_points(0.5), ns.posterior_points(0.5))


def test_prior_points():
    ns = NestedSamples(root='./tests/example_data/pc')
    assert_array_equal(ns.prior_points(), ns.posterior_points(0))


def test_NestedSamples_importance_sample():
    np.random.seed(3)
    ns0 = NestedSamples(root='./tests/example_data/pc')
    pi0 = ns0.set_beta(0)
    NS0 = ns0.ns_output(nsamples=2000)

    with pytest.raises(NotImplementedError):
        ns0.importance_sample(ns0.logL, action='spam')

    ns_masked = ns0.importance_sample(ns0.logL, action='replace')
    assert_array_equal(ns0.logL, ns_masked.logL)
    assert_array_equal(ns0.logL_birth, ns_masked.logL_birth)
    assert_array_equal(ns0.weights, ns_masked.weights)

    ns_masked = ns0.importance_sample(np.zeros_like(ns0.logL), action='add')
    assert_array_equal(ns0.logL, ns_masked.logL)
    assert_array_equal(ns0.logL_birth, ns_masked.logL_birth)
    assert_array_equal(ns0.weights, ns_masked.weights)

    mask = ((ns0.x0 > -0.3) & (ns0.x2 > 0.2) & (ns0.x4 < 3.5)).to_numpy()
    ns_masked = merge_nested_samples((ns0[mask], ))
    V_prior = pi0[mask].weights.sum() / pi0.weights.sum()
    V_posterior = ns0[mask].weights.sum() / ns0.weights.sum()

    ns1 = ns0.importance_sample(mask, action='mask')
    assert_array_equal(ns_masked.logL, ns1.logL)
    assert_array_equal(ns_masked.logL_birth, ns1.logL_birth)
    assert_array_equal(ns_masked.weights, ns1.weights)

    logL_new = np.where(mask, 0, -np.inf)
    ns1 = ns0.importance_sample(logL_new)
    NS1 = ns1.ns_output(nsamples=2000)
    assert_array_equal(ns1, ns_masked)
    logZ_V = NS0.logZ.mean() + np.log(V_posterior) - np.log(V_prior)
    assert abs(NS1.logZ.mean() - logZ_V) < 1.5 * NS1.logZ.std()

    logL_new = np.where(mask, 0, -1e30)
    ns1 = ns0.importance_sample(logL_new)
    NS1 = ns1.ns_output(nsamples=2000)
    logZ_V = NS0.logZ.mean() + np.log(V_posterior)
    assert abs(NS1.logZ.mean() - logZ_V) < 1.5 * NS1.logZ.std()

    ns0.importance_sample(logL_new, inplace=True)
    assert type(ns0) is NestedSamples
    assert_array_equal(ns0, ns1)
    assert ns0.tex == ns1.tex
    assert ns0.root == ns1.root
    assert ns0.label == ns1.label
    assert ns0.beta == ns1.beta
    assert ns0 is not ns1
    assert ns0.tex is not ns1.tex


def test_MCMCSamples_importance_sample():
    np.random.seed(3)
    mc0 = MCMCSamples(root='./tests/example_data/gd')

    with pytest.raises(NotImplementedError):
        mc0.importance_sample(mc0.logL, action='spam')

    # new gaussian logL
    logL_i = norm.logpdf(mc0.x3, loc=0.4, scale=0.1)

    # add logL
    mc1 = mc0.importance_sample(np.zeros_like(mc0.logL), action='add')
    assert_array_equal(mc0.logL, mc1.logL)
    assert_array_equal(mc0.weights, mc1.weights)
    mc1 = mc0.importance_sample(logL_new=logL_i)
    assert np.all(mc1.logL.to_numpy() != mc0.logL.to_numpy())
    assert not np.all(mc1.weights == mc0.weights)

    # replace logL
    mc2 = mc0.importance_sample(mc0.logL, action='replace')
    assert_array_equal(mc0.logL, mc2.logL)
    assert_array_equal(mc0.weights, mc2.weights)
    mc2 = mc0.importance_sample(mc0.logL.to_numpy()+logL_i, action='replace')
    assert np.all(mc2.logL.to_numpy() != mc0.logL.to_numpy())
    assert not np.all(mc2.weights == mc0.weights)
    assert_array_equal(mc1.logL.to_numpy(), mc2.logL.to_numpy())
    assert_array_almost_equal(mc1.logL.to_numpy(), mc2.logL.to_numpy())

    # mask logL
    mask = ((mc0.x0 > -0.3) & (mc0.x2 > 0.2) & (mc0.x4 < 3.5)).to_numpy()
    mc_masked = mc0[mask]
    mc3 = mc0.importance_sample(mask, action='mask')
    assert_array_equal(mc_masked.logL, mc3.logL)
    assert_array_equal(mc_masked.weights, mc3.weights)
    assert np.all(mc3.x0 > -0.3)

    for mc in [mc1, mc2, mc3]:
        assert mc.tex == mc0.tex
        assert mc.root == mc0.root
        assert mc.label == mc0.label
        assert mc._metadata == mc0._metadata
        assert mc is not mc0
        assert mc.tex is not mc0.tex

    mc0.importance_sample(mask, action='mask', inplace=True)
    assert type(mc0) is MCMCSamples
    assert_array_equal(mc3, mc0)
    assert mc3.tex == mc0.tex
    assert mc3.root == mc0.root
    assert mc3.label == mc0.label
    assert mc3._metadata == mc0._metadata
    assert mc3 is not mc0
    assert mc3.tex is not mc0.tex


def test_wedding_cake():
    np.random.seed(3)
    wc = WeddingCake(4, 0.5, 0.01)
    nlive = 500
    samples = wc.sample(nlive)
    assert samples.nlive.iloc[0] == nlive
    assert samples.nlive.iloc[-1] == 1
    assert (samples.nlive <= nlive).all()
    out = samples.logZ(100)
    assert abs(out.mean()-wc.logZ()) < out.std()*3


def test_logzero_mask_prior_level():
    np.random.seed(3)
    ns0 = NestedSamples(root='./tests/example_data/pc')
    pi0 = ns0.set_beta(0)
    NS0 = ns0.ns_output(nsamples=2000)
    mask = ((ns0.x0 > -0.3) & (ns0.x2 > 0.2) & (ns0.x4 < 3.5)).to_numpy()

    V_prior = pi0[mask].weights.sum() / pi0.weights.sum()
    V_posterior = ns0[mask].weights.sum() / ns0.weights.sum()
    logZ_V = NS0.logZ.mean() + np.log(V_posterior) - np.log(V_prior)

    ns1 = merge_nested_samples((ns0[mask],))
    NS1 = ns1.ns_output(nsamples=2000)

    assert abs(NS1.logZ.mean() - logZ_V) < 1.5 * NS1.logZ.std()


def test_logzero_mask_likelihood_level():
    np.random.seed(3)
    ns0 = NestedSamples(root='./tests/example_data/pc')
    NS0 = ns0.ns_output(nsamples=2000)
    mask = ((ns0.x0 > -0.3) & (ns0.x2 > 0.2) & (ns0.x4 < 3.5)).to_numpy()

    V_posterior = ns0[mask].weights.sum() / ns0.weights.sum()
    logZ_V = NS0.logZ.mean() + np.log(V_posterior)

    ns1 = NestedSamples(root='./tests/example_data/pc')
    ns1.logL = np.where(mask, ns1.logL, -1e30)
    ns1 = merge_nested_samples((ns1[ns1.logL > ns1.logL_birth],))
    NS1 = ns1.ns_output(nsamples=2000)

    assert abs(NS1.logZ.mean() - logZ_V) < 1.5 * NS1.logZ.std()


def test_recompute():
    np.random.seed(3)
    pc = NestedSamples(root='./tests/example_data/pc')
    recompute = pc.recompute()
    assert recompute is not pc

    pc.loc[1000, 'logL'] = pc.logL_birth.iloc[1000]-1
    with pytest.warns(RuntimeWarning):
        recompute = pc.recompute()
    assert len(recompute) == len(pc) - 1

    mn = NestedSamples(root='./tests/example_data/mn_old')
    with pytest.raises(RuntimeError):
        mn.recompute()


def test_NaN():
    np.random.seed(3)
    pc = NestedSamples(root='./tests/example_data/pc')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pc_new = pc.copy()
        pc_new.loc[2, "logL"] = np.nan
        pc_new.recompute(inplace=True)
        assert len(w) == 1
        assert "NaN encountered in logL." in str(w[-1].message)
        assert len(pc_new) == len(pc) - 1
        assert pc_new.nlive.iloc[0] == 124


def test_unsorted():
    np.random.seed(4)
    pc = NestedSamples(root='./tests/example_data/pc')
    i = np.random.choice(len(pc), len(pc), replace=False)
    pc_resort = NestedSamples(data=pc.loc[i, ['x0', 'x1', 'x2', 'x3', 'x4']],
                              logL=pc.loc[i, 'logL'],
                              logL_birth=pc.loc[i, 'logL_birth'])
    assert_array_equal(pc_resort, pc)


def test_copy():
    np.random.seed(3)
    pc = NestedSamples(root='./tests/example_data/pc')
    new = pc.copy()
    assert new is not pc
    assert new.tex is not pc.tex


def test_plotting_with_integer_names():
    np.random.seed(3)
    samples_1 = Samples(data=np.random.rand(1000, 3))
    samples_2 = Samples(data=np.random.rand(1000, 3))
    samples_1.compress()
    fig, ax = samples_1.plot_2d([0, 1, 2])
    samples_2.plot_2d(ax)

    fig, ax = samples_1.plot_1d([0, 1, 2])
    samples_2.plot_1d(ax)

    assert samples_1[0].shape == (1000,)
    assert_array_equal(samples_1.loc[:, 0], samples_1[0])
    assert_array_equal(samples_1.loc[:, 0], samples_1.iloc[:, 0])
    with pytest.raises(KeyError):
        samples_1['0']


def test_logL_list():
    np.random.seed(5)
    default = NestedSamples(root='./tests/example_data/pc')
    logL = default.logL.tolist()
    logL_birth = default.logL_birth.tolist()
    data = default.iloc[:, :5].to_numpy().tolist()

    samples = NestedSamples(data=data, logL=logL, logL_birth=logL_birth)
    assert_array_equal(default, samples)
    plt.close("all")


def test_samples_dot_plot():
    samples = NestedSamples(root='./tests/example_data/pc')
    axes = samples[['x0', 'x1', 'x2', 'x3', 'x4']].plot.hist()
    assert len(axes.containers) == 5
    axes = samples.x0.plot.kde(subplots=True)
    assert len(axes) == 1
    axes = samples[['x0', 'x1']].plot.kde(subplots=True)
    assert len(axes) == 2
    plt.close("all")

    axes = samples.plot.kde_2d('x0', 'x1')
    assert len(axes.collections) == 5
    assert axes.get_xlabel() == 'x0'
    assert axes.get_ylabel() == 'x1'
    plt.close("all")
    axes = samples.plot.hist_2d('x1', 'x0')
    assert len(axes.collections) == 1
    assert axes.get_xlabel() == 'x1'
    assert axes.get_ylabel() == 'x0'
    plt.close("all")
    axes = samples.plot.scatter_2d('x2', 'x3')
    assert len(axes.lines) == 1
    plt.close("all")
    axes = samples.x1.plot.kde_1d()
    assert len(axes.lines) == 1
    plt.close("all")
    axes = samples.x2.plot.hist_1d()
    assert len(axes.containers) == 1
    plt.close("all")

    try:
        axes = samples.plot.fastkde_2d('x0', 'x1')
        assert len(axes.collections) == 5
        plt.close("all")
        axes = samples.plot.fastkde_1d()
        assert len(axes.lines) == 1
        plt.close("all")
    except ImportError:
        pass

    plt.close("all")


def test_samples_plot_labels():
    samples = NestedSamples(root='./tests/example_data/pc')
    columns = ['x0', 'x1', 'x2', 'x3', 'x4']
    fig, axes = samples.plot_2d(columns)

    for col, ax in zip(columns, axes.loc[:, 'x0']):
        assert samples.tex[col] == ax.get_ylabel()

    for col, ax in zip(columns, axes.loc['x4', :]):
        assert samples.tex[col] == ax.get_xlabel()
