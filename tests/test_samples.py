import anesthetic.examples._matplotlib_agg  # noqa: F401

import pytest
from contextlib import nullcontext
from math import floor, ceil
import numpy as np
from pandas import MultiIndex
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from anesthetic.weighted_pandas import WeightedSeries, WeightedDataFrame
from anesthetic import (
    Samples, MCMCSamples, NestedSamples, make_1d_axes, make_2d_axes,
    read_chains
)
from anesthetic.samples import (merge_nested_samples, merge_samples_weighted,
                                WeightedLabelledSeries,
                                WeightedLabelledDataFrame)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_array_less, assert_allclose)
from pandas.testing import assert_frame_equal
from matplotlib.colors import to_hex
from scipy.stats import ks_2samp, kstest, norm
from utils import skipif_no_fastkde, astropy_mark_xfail, fastkde_mark_skip


@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    yield
    plt.close("all")


def test_build_samples():
    np.random.seed(3)
    nsamps = 1000
    ndims = 3
    data = np.random.randn(nsamps, ndims)
    logL = np.random.rand(nsamps)
    weights = np.random.randint(1, 20, size=nsamps)
    params = ['A', 'B', 'C']
    labels = {'A': '$A$', 'B': '$B$', 'C': '$C$'}
    labels = [labels.get(p, p) for p in params]

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

    s = Samples(data=data, columns=params, labels=labels)

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
    assert not hasattr(mc, 'root')
    assert not hasattr(ns, 'root')


def test_different_parameters():
    np.random.seed(3)
    params_x = ['x0', 'x1', 'x2', 'x3', 'x4']
    params_y = ['x0', 'x1', 'x2']
    fig, axes = make_1d_axes(params_x)
    ns = read_chains('./tests/example_data/pc')
    ns.plot_1d(axes)
    fig, axes = make_2d_axes(params_y)
    ns.plot_2d(axes)
    fig, axes = make_2d_axes(params_x)
    ns.plot_2d(axes)
    fig, axes = make_2d_axes([params_x, params_y])
    ns.plot_2d(axes)


def test_manual_columns():
    old_params = ['x0', 'x1', 'x2', 'x3', 'x4']
    mcmc_params = ['logL', 'chain']
    ns_params = ['logL', 'logL_birth', 'nlive']
    mcmc = read_chains('./tests/example_data/gd')
    ns = read_chains('./tests/example_data/pc')
    assert_array_equal(mcmc.drop_labels().columns, old_params + mcmc_params)
    assert_array_equal(ns.drop_labels().columns, old_params + ns_params)

    new_params = ['y0', 'y1', 'y2', 'y3', 'y4']
    mcmc = read_chains('./tests/example_data/gd', columns=new_params)
    ns = read_chains('./tests/example_data/pc', columns=new_params)
    assert_array_equal(mcmc.drop_labels().columns, new_params + mcmc_params)
    assert_array_equal(ns.drop_labels().columns, new_params + ns_params)


def test_plot_2d_kinds():
    np.random.seed(3)
    ns = read_chains('./tests/example_data/pc')
    params_x = ['x0', 'x1', 'x2', 'x3']
    params_y = ['x0', 'x1', 'x2']
    params = [params_x, params_y]

    # Check dictionaries
    axes = ns.plot_2d(params, kind={'lower': 'kde_2d'})
    assert (~axes.isnull()).to_numpy().sum() == 3

    axes = ns.plot_2d(params, kind={'upper': 'scatter_2d'})
    assert (~axes.isnull()).to_numpy().sum() == 6

    axes = ns.plot_2d(params, kind={'upper': 'kde_2d', 'diagonal': 'kde_1d'})
    assert (~axes.isnull()).to_numpy().sum() == 9

    axes = ns.plot_2d(params, kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'})
    assert (~axes.isnull()).to_numpy().sum() == 6

    axes = ns.plot_2d(params, kind={'lower': 'kde_2d', 'diagonal': 'kde_1d'})
    assert (~axes.isnull()).to_numpy().sum() == 6

    axes = ns.plot_2d(params, kind={'lower': 'kde_2d',
                                    'diagonal': 'kde_1d',
                                    'upper': 'scatter_2d'})
    assert (~axes.isnull()).to_numpy().sum() == 12

    # Check strings
    axes = ns.plot_2d(params, kind='kde')
    assert (~axes.isnull()).to_numpy().sum() == 6
    axes = ns.plot_2d(params, kind='kde_1d')
    assert (~axes.isnull()).to_numpy().sum() == 3
    axes = ns.plot_2d(params, kind='kde_2d')
    assert (~axes.isnull()).to_numpy().sum() == 3

    # Check kinds vs kind kwarg
    axes = ns.plot_2d(params, kinds='kde')
    assert (~axes.isnull()).to_numpy().sum() == 6
    axes = ns.plot_2d(params, kinds='kde_1d')
    assert (~axes.isnull()).to_numpy().sum() == 3
    axes = ns.plot_2d(params, kinds='kde_2d')
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


def test_plot_2d_kinds_multiple_calls():
    np.random.seed(3)
    ns = read_chains('./tests/example_data/pc')
    params = ['x0', 'x1', 'x2', 'x3']

    axes = ns.plot_2d(params, kind={'diagonal': 'kde_1d',
                                    'lower': 'kde_2d',
                                    'upper': 'scatter_2d'})
    ns.plot_2d(axes, kind={'diagonal': 'hist_1d'})

    axes = ns.plot_2d(params, kind={'diagonal': 'hist_1d'})
    ns.plot_2d(axes, kind={'diagonal': 'kde_1d',
                           'lower': 'kde_2d',
                           'upper': 'scatter_2d'})


def test_root_and_label():
    np.random.seed(3)
    ns = read_chains('./tests/example_data/pc')
    assert ns.root == './tests/example_data/pc'
    assert ns.label == 'pc'

    ns = NestedSamples()
    assert not hasattr(ns, 'root')
    assert ns.label is None

    mc = read_chains('./tests/example_data/gd')
    assert (mc.root == './tests/example_data/gd')
    assert mc.label == 'gd'

    mc = MCMCSamples()
    assert not hasattr(mc, 'root')
    assert mc.label is None


def test_plot_2d_legend():
    np.random.seed(3)
    ns = read_chains('./tests/example_data/pc')
    mc = read_chains('./tests/example_data/gd')
    params = ['x0', 'x1', 'x2', 'x3']

    # Test label kwarg for kde
    fig, axes = make_2d_axes(params, upper=False)
    ns.plot_2d(axes, label='l1', kind=dict(diagonal='kde_1d', lower='kde_2d'))
    mc.plot_2d(axes, label='l2', kind=dict(diagonal='kde_1d', lower='kde_2d'))

    for y, row in axes.iterrows():
        for x, ax in row.items():
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

    # Test label kwarg for hist and scatter
    fig, axes = make_2d_axes(params, lower=False)
    ns.plot_2d(axes, label='l1', kind=dict(diagonal='hist_1d',
                                           upper='scatter_2d'))
    mc.plot_2d(axes, label='l2', kind=dict(diagonal='hist_1d',
                                           upper='scatter_2d'))

    for y, row in axes.iterrows():
        for x, ax in row.items():
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

    # test default labelling
    fig, axes = make_2d_axes(params, upper=False)
    ns.plot_2d(axes)
    mc.plot_2d(axes)

    for y, row in axes.iterrows():
        for x, ax in row.items():
            if ax is not None:
                handles, labels = ax.get_legend_handles_labels()
                assert labels == ['pc', 'gd']

    # Test label kwarg to constructors
    ns = read_chains('./tests/example_data/pc', label='l1')
    mc = read_chains('./tests/example_data/gd', label='l2')
    params = ['x0', 'x1', 'x2', 'x3']

    fig, axes = make_2d_axes(params, upper=False)
    ns.plot_2d(axes)
    mc.plot_2d(axes)

    for y, row in axes.iterrows():
        for x, ax in row.items():
            if ax is not None:
                handles, labels = ax.get_legend_handles_labels()
                assert labels == ['l1', 'l2']


@pytest.mark.parametrize('kind', ['kde', 'hist', skipif_no_fastkde('fastkde')])
def test_plot_2d_colours(kind):
    np.random.seed(3)
    gd = read_chains("./tests/example_data/gd")
    gd.drop(columns='x3', inplace=True, level=0)
    pc = read_chains("./tests/example_data/pc")
    pc.drop(columns='x4', inplace=True, level=0)
    mn = read_chains("./tests/example_data/mn")
    mn.drop(columns='x2', inplace=True, level=0)

    fig = plt.figure()
    fig, axes = make_2d_axes(['x0', 'x1', 'x2', 'x3', 'x4'], fig=fig)
    kinds = {'diagonal': kind + '_1d',
             'lower': kind + '_2d',
             'upper': 'scatter_2d'}
    gd.plot_2d(axes, kind=kinds, label="gd")
    pc.plot_2d(axes, kind=kinds, label="pc")
    mn.plot_2d(axes, kind=kinds, label="mn")
    gd_colors = []
    pc_colors = []
    mn_colors = []
    for y, rows in axes.iterrows():
        for x, ax in rows.items():
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


@pytest.mark.parametrize('kwargs', [dict(color='r', alpha=0.5, ls=':', lw=1),
                                    dict(c='r', linestyle=':', linewidth=1),
                                    dict(ec='r', fc='b'),
                                    dict(edgecolor='r', facecolor='b'),
                                    dict(cmap=plt.cm.RdBu),
                                    dict(colormap=plt.cm.RdBu),
                                    dict(cmap="viridis"),
                                    dict(colormap="viridis")])
@pytest.mark.parametrize('kind', ['kde', 'hist', 'default',
                                  skipif_no_fastkde('fastkde')])
def test_plot_2d_kwargs(kind, kwargs):
    np.random.seed(42)
    pc = read_chains("./tests/example_data/pc")
    fig, axes = make_2d_axes(['x0', 'x1'])
    pc.plot_2d(axes, kind=kind, **kwargs)


@pytest.mark.parametrize('kind', ['kde', 'hist', skipif_no_fastkde('fastkde')])
def test_plot_1d_colours(kind):
    np.random.seed(3)
    gd = read_chains("./tests/example_data/gd")
    gd.drop(columns='x3', inplace=True, level=0)
    pc = read_chains("./tests/example_data/pc")
    pc.drop(columns='x4', inplace=True, level=0)
    mn = read_chains("./tests/example_data/mn")
    mn.drop(columns='x2', inplace=True, level=0)

    fig = plt.figure()
    fig, axes = make_1d_axes(['x0', 'x1', 'x2', 'x3', 'x4'], fig=fig)
    gd.plot_1d(axes, kind=kind + '_1d', label="gd")
    pc.plot_1d(axes, kind=kind + '_1d', label="pc")
    mn.plot_1d(axes, kind=kind + '_1d', label="mn")
    gd_colors = []
    pc_colors = []
    mn_colors = []
    for x, ax in axes.items():
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


@astropy_mark_xfail
def test_astropyhist():
    np.random.seed(3)
    ns = read_chains('./tests/example_data/pc')
    with pytest.raises(ValueError):
        ns.plot_1d(['x0', 'x1', 'x2', 'x3'], kind='hist_1d', bins='knuth')


def test_hist_levels():
    np.random.seed(3)
    ns = read_chains('./tests/example_data/pc')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'], kind={'lower': 'hist_2d'},
               levels=[0.95, 0.68], bins=20)


def test_plot_2d_no_axes():
    np.random.seed(3)
    ns = read_chains('./tests/example_data/pc')
    axes = ns[['x0', 'x1', 'x2']].plot_2d()
    assert axes.iloc[-1, 0].get_xlabel() == '$x_0$'
    assert axes.iloc[-1, 1].get_xlabel() == '$x_1$'
    assert axes.iloc[-1, 2].get_xlabel() == '$x_2$'

    axes = ns[['x0', 'x1', 'x2']].drop_labels().plot_2d()
    assert axes.iloc[-1, 0].get_xlabel() == 'x0'
    assert axes.iloc[-1, 1].get_xlabel() == 'x1'
    assert axes.iloc[-1, 2].get_xlabel() == 'x2'


def test_plot_1d_no_axes():
    np.random.seed(3)
    ns = read_chains('./tests/example_data/pc')
    axes = ns[['x0', 'x1', 'x2']].plot_1d()
    assert axes.iloc[0].get_xlabel() == '$x_0$'
    assert axes.iloc[1].get_xlabel() == '$x_1$'
    assert axes.iloc[2].get_xlabel() == '$x_2$'
    axes = ns[['x0', 'x1', 'x2']].drop_labels().plot_1d()
    assert axes.iloc[0].get_xlabel() == 'x0'
    assert axes.iloc[1].get_xlabel() == 'x1'
    assert axes.iloc[2].get_xlabel() == 'x2'


@pytest.mark.parametrize('kind', ['kde', 'hist', skipif_no_fastkde('fastkde')])
def test_plot_logscale_1d(kind):
    ns = read_chains('./tests/example_data/pc')
    params = ['x0', 'x1', 'x2', 'x3', 'x4']

    # 1d
    axes = ns.plot_1d(params, kind=kind + '_1d', logx=['x2'])
    for x, ax in axes.items():
        if x == 'x2':
            assert ax.get_xscale() == 'log'
        else:
            assert ax.get_xscale() == 'linear'
    ax = axes.loc['x2']
    if 'kde' in kind:
        p = ax.get_children()
        arg = np.argmax(p[0].get_ydata())
        pmax = np.log10(p[0].get_xdata()[arg])
        d = 0.1
    else:
        arg = np.argmax([p.get_height() for p in ax.patches])
        pmax = np.log10(ax.patches[arg].get_x())
        d = np.log10(ax.patches[arg+1].get_x() / ax.patches[arg].get_x())
    assert pmax == pytest.approx(-1, abs=d)


@pytest.mark.parametrize('kind', ['kde', 'hist', skipif_no_fastkde('fastkde')])
def test_plot_logscale_2d(kind):
    ns = read_chains('./tests/example_data/pc')
    params = ['x0', 'x1', 'x2', 'x3', 'x4']

    # 2d, logx only
    axes = ns.plot_2d(params, kind=kind, logx=['x2'])
    for y, rows in axes.iterrows():
        for x, ax in rows.items():
            if ax is not None:
                if x == 'x2':
                    assert ax.get_xscale() == 'log'
                else:
                    assert ax.get_xscale() == 'linear'
                ax.get_yscale() == 'linear'
                if x == y:
                    if x == 'x2':
                        assert ax.twin.get_xscale() == 'log'
                    else:
                        assert ax.twin.get_xscale() == 'linear'
                    assert ax.twin.get_yscale() == 'linear'

    # 2d, logy only
    axes = ns.plot_2d(params, kind=kind, logy=['x2'])
    for y, rows in axes.iterrows():
        for x, ax in rows.items():
            if ax is not None:
                ax.get_xscale() == 'linear'
                if y == 'x2':
                    assert ax.get_yscale() == 'log'
                else:
                    assert ax.get_yscale() == 'linear'
                if x == y:
                    assert ax.twin.get_xscale() == 'linear'
                    assert ax.twin.get_yscale() == 'linear'

    # 2d, logx and logy
    axes = ns.plot_2d(params, kind=kind, logx=['x2'], logy=['x2'])
    for y, rows in axes.iterrows():
        for x, ax in rows.items():
            if ax is not None:
                if x == 'x2':
                    assert ax.get_xscale() == 'log'
                else:
                    assert ax.get_xscale() == 'linear'
                if y == 'x2':
                    assert ax.get_yscale() == 'log'
                else:
                    assert ax.get_yscale() == 'linear'
                if x == y:
                    if x == 'x2':
                        assert ax.twin.get_xscale() == 'log'
                    else:
                        assert ax.twin.get_xscale() == 'linear'
                    assert ax.twin.get_yscale() == 'linear'


@pytest.mark.parametrize('k', ['hist_1d', 'hist'])
@pytest.mark.parametrize('b', ['scott', 10, np.logspace(-3, 0, 20)])
@pytest.mark.parametrize('r', [None, (1e-5, 1)])
def test_plot_logscale_hist_kwargs(k, b, r):
    ns = read_chains('./tests/example_data/pc')
    with pytest.warns(UserWarning) if k == 'hist' else nullcontext():
        axes = ns[['x2']].plot_1d(kind=k, logx=['x2'], bins=b, range=r)
    ax = axes.loc['x2']
    assert ax.get_xscale() == 'log'
    arg = np.argmax([p.get_height() for p in ax.patches])
    pmax = np.log10(ax.patches[arg].get_x())
    d = np.log10(ax.patches[arg+1].get_x() / ax.patches[arg].get_x())
    assert pmax == pytest.approx(-1, abs=d)


def test_logscale_failure_without_match():
    ns = read_chains('./tests/example_data/pc')
    params = ['x0', 'x2']

    # 1d
    axes = ns.plot_1d(params)
    with pytest.raises(ValueError):
        ns.plot_1d(axes, logx=['x2'])
    fig, axes = make_1d_axes(params)
    with pytest.raises(ValueError):
        ns.plot_1d(axes, logx=['x2'])

    # 2d
    axes = ns.plot_2d(params)
    with pytest.raises(ValueError):
        ns.plot_2d(axes, logx=['x2'])
    axes = ns.plot_2d(params)
    with pytest.raises(ValueError):
        ns.plot_2d(axes, logy=['x2'])
    axes = ns.plot_2d(params)
    with pytest.raises(ValueError):
        ns.plot_2d(axes, logx=['x2'], logy=['x2'])
    fig, axes = make_2d_axes(params)
    with pytest.raises(ValueError):
        ns.plot_2d(axes, logx=['x2'])
    fig, axes = make_2d_axes(params)
    with pytest.raises(ValueError):
        ns.plot_2d(axes, logy=['x2'])
    fig, axes = make_2d_axes(params)
    with pytest.raises(ValueError):
        ns.plot_2d(axes, logx=['x2'], logy=['x2'])


def test_mcmc_stats():
    mcmc = read_chains('./tests/example_data/cb')
    chains = mcmc.groupby(('chain', '$n_\\mathrm{chain}$'), group_keys=False)
    n0, n1 = chains.count().iloc[:, 0]  # number samples in first chain
    mcmc_head = chains.head(200).copy()
    mcmc_tail = mcmc.remove_burn_in(burn_in=200)
    mcmc_half = mcmc.remove_burn_in(burn_in=0.5)

    # check indices after burn-in removal
    assert mcmc_tail.index.get_level_values(0)[0] == 200
    assert mcmc_tail.index.get_level_values(0)[n0] == 200 + n0 + 200
    assert mcmc_half.index.get_level_values(0)[0] == floor(n0/2)
    assert mcmc_half.index.get_level_values(0)[ceil(n0/2)] == n0 + floor(n1/2)

    # check Gelman--Rubin statistic
    assert mcmc_head.Gelman_Rubin() > 0.1
    assert mcmc_tail.Gelman_Rubin() < 0.01
    assert mcmc_half.Gelman_Rubin() < 0.01
    assert mcmc_half.Gelman_Rubin(['x0']) < 0.01
    assert mcmc_half.Gelman_Rubin(['x1']) < 0.01
    with pytest.raises(np.linalg.LinAlgError):
        mcmc['y1'] = mcmc.x1
        mcmc['y2'] = mcmc.x1
        mcmc['y3'] = mcmc.x1
        mcmc.Gelman_Rubin(['x0', 'x1', 'y1', 'y2', 'y3'])

    # check per-parameter Gelman--Rubin statistic
    GR_par = mcmc_head.Gelman_Rubin(per_param='par')
    GR_cov = mcmc_head.Gelman_Rubin(per_param='cov')
    assert_array_equal(np.ravel(GR_par), np.diag(GR_cov))
    assert np.all(GR_par > 0.1)
    assert np.all(GR_cov > 0.1)
    GR_par = mcmc_tail.Gelman_Rubin(per_param='par')
    GR_cov = mcmc_tail.Gelman_Rubin(per_param='cov')
    assert_array_equal(np.ravel(GR_par), np.diag(GR_cov))
    assert np.all(GR_par < 0.01)
    assert np.all(GR_cov < 0.01)
    GR_par = mcmc_half.Gelman_Rubin(per_param='par')
    GR_cov = mcmc_half.Gelman_Rubin(per_param='cov')
    assert_array_equal(np.ravel(GR_par), np.diag(GR_cov))
    assert np.all(GR_par < 0.01)
    assert np.all(GR_cov < 0.01)
    assert len(mcmc_half.Gelman_Rubin(per_param=True)) == 2
    assert len(mcmc_half.Gelman_Rubin(per_param='all')) == 2
    assert_array_equal(mcmc_half.Gelman_Rubin(per_param=True)[1], GR_par)
    assert_array_equal(mcmc_half.Gelman_Rubin(per_param='all')[1], GR_cov)

    # more burn-in checks
    mcmc_new = mcmc.remove_burn_in(burn_in=200.9)
    assert len(mcmc_new) == n0 - 200 + n1 - 200
    assert mcmc_new.index.get_level_values(0)[0] == 200
    assert mcmc_new.index.get_level_values(0)[n0] == 200 + n0 + 200
    mcmc_new = mcmc.remove_burn_in(burn_in=-0.5)
    assert len(mcmc_new) == floor(n0/2) + floor(n1/2)
    assert mcmc_new.index.get_level_values(0)[0] == ceil(n0/2)
    assert mcmc_new.index.get_level_values(0)[floor(n0/2)] == n0 + floor(n1/2)
    mcmc_new = mcmc.remove_burn_in(burn_in=-200)
    assert len(mcmc_new) == 200 + 200
    assert mcmc_new.index.get_level_values(0)[0] == n0 - 200
    assert mcmc_new.index.get_level_values(0)[200] == n0 + n1 - 200
    mcmc_new = mcmc.remove_burn_in(burn_in=[0.8, -0.75])
    assert len(mcmc_new) == ceil(n0/5) + floor(3*n1/4)
    assert mcmc_new.index.get_level_values(0)[0] == floor(4*n0/5)
    assert mcmc_new.index.get_level_values(0)[ceil(n0/5)] == n0 + ceil(n1/4)
    mcmc_new = mcmc.remove_burn_in(burn_in=[2, -100])
    assert len(mcmc_new) == n0 - 2 + 100
    assert mcmc_new.index.get_level_values(0)[0] == 2
    assert mcmc_new.index.get_level_values(0)[n0-2] == n0 + n1 - 100

    # test reset index
    mcmc_new = mcmc.remove_burn_in(burn_in=200, reset_index=True)
    assert len(mcmc_new) == n0 - 200 + n1 - 200
    assert mcmc_new.index.get_level_values(0)[0] == 0
    assert mcmc_new.index.get_level_values(0)[-1] == n0 - 200 + n1 - 200 - 1

    # test inplace
    assert mcmc.index.get_level_values(0)[0] == 0
    assert mcmc.index.get_level_values(0)[n0] == n0
    mcmc_new = mcmc.remove_burn_in(burn_in=200, inplace=True)
    assert mcmc_new is None
    assert len(mcmc) == n0 - 200 + n1 - 200
    assert mcmc.index.get_level_values(0)[0] == 200
    assert mcmc.index.get_level_values(0)[n0] == 200 + n0 + 200

    with pytest.raises(ValueError):
        mcmc.remove_burn_in(burn_in=[1, 2, 3])


def test_logX():
    np.random.seed(3)
    pc = read_chains('./tests/example_data/pc')

    logX = pc.logX()
    assert isinstance(logX, WeightedSeries)
    assert_array_equal(logX.index, pc.index)

    nsamples = 10

    logX = pc.logX(nsamples=nsamples)
    assert isinstance(logX, WeightedDataFrame)
    assert_array_equal(logX.index, pc.index)
    assert_array_equal(logX.columns, np.arange(nsamples))
    assert logX.columns.name == 'samples'

    assert not (logX.diff(axis=0) > 0).to_numpy().any()

    n = 1000
    logX = pc.logX(n)

    assert (abs(logX.mean(axis=1) - pc.logX()) < logX.std(axis=1) * 3).all()


def test_logdX():
    np.random.seed(3)
    pc = read_chains('./tests/example_data/pc')

    logdX = pc.logdX()
    assert isinstance(logdX, WeightedSeries)
    assert_array_equal(logdX.index, pc.index)

    nsamples = 10

    logdX = pc.logdX(nsamples=nsamples)
    assert isinstance(logdX, WeightedDataFrame)
    assert_array_equal(logdX.index, pc.index)
    assert_array_equal(logdX.columns, np.arange(nsamples))
    assert logdX.columns.name == 'samples'

    assert not (logdX > 0).to_numpy().any()

    n = 1000
    logdX = pc.logdX(n)

    assert (abs(logdX.mean(axis=1) - pc.logdX()) < logdX.std(axis=1) * 3).all()


def test_logbetaL():
    np.random.seed(3)
    pc = read_chains('./tests/example_data/pc')

    logX = pc.logX()
    assert isinstance(logX, WeightedSeries)
    assert_array_equal(logX.index, pc.index)

    nsamples = 10

    logX = pc.logX(nsamples=nsamples)
    assert isinstance(logX, WeightedDataFrame)
    assert_array_equal(logX.index, pc.index)
    assert_array_equal(logX.columns, np.arange(nsamples))
    assert logX.columns.name == 'samples'

    assert not (logX.diff(axis=0) > 0).to_numpy().any()

    n = 1000
    logX = pc.logX(n)

    assert (abs(logX.mean(axis=1) - pc.logX()) < logX.std(axis=1) * 3).all()


def test_logw():
    np.random.seed(3)
    pc = read_chains('./tests/example_data/pc')

    logw = pc.logw()
    assert isinstance(logw, WeightedSeries)
    assert_array_equal(logw.index, pc.index)

    nsamples = 10
    beta = [0., 0.5, 1.]

    logw = pc.logw(nsamples=nsamples)
    assert isinstance(logw, WeightedDataFrame)
    assert_array_equal(logw.index, pc.index)
    assert logw.columns.name == 'samples'
    assert_array_equal(logw.columns, range(nsamples))

    logw = pc.logw(beta=beta)
    assert isinstance(logw, WeightedDataFrame)
    assert_array_equal(logw.index, pc.index)
    assert logw.columns.name == 'beta'
    assert_array_equal(logw.columns, beta)

    logw = pc.logw(nsamples=nsamples, beta=beta)
    assert isinstance(logw, WeightedDataFrame)
    assert logw.columns.names == ['beta', 'samples']
    assert logw.columns.levshape == (len(beta), nsamples)

    n = 1000
    logw = pc.logw(n)

    assert (abs(logw.mean(axis=1) - pc.logw()) < logw.std(axis=1) * 3).all()


def test_logZ():
    np.random.seed(3)
    pc = read_chains('./tests/example_data/pc')

    logZ = pc.logZ()
    assert isinstance(logZ, float)

    nsamples = 10
    beta = [0., 0.5, 1.]

    logZ = pc.logZ(nsamples=nsamples)
    assert isinstance(logZ, WeightedLabelledSeries)
    assert logZ.index.name == 'samples'
    assert logZ.name == 'logZ'
    assert_array_equal(logZ.index, range(nsamples))

    logZ = pc.logZ(beta=beta)
    assert isinstance(logZ, WeightedLabelledSeries)
    assert logZ.index.name == 'beta'
    assert logZ.name == 'logZ'
    assert len(logZ) == len(beta)

    logZ = pc.logZ(nsamples=nsamples, beta=beta)
    assert isinstance(logZ, WeightedLabelledSeries)
    assert logZ.index.names == ['beta', 'samples']
    assert logZ.name == 'logZ'
    assert logZ.index.levshape == (len(beta), nsamples)

    n = 1000
    logZ = pc.logZ(n)

    assert abs(logZ.mean() - pc.logZ()) < logZ.std() * 3


def test_D_KL():
    np.random.seed(3)
    pc = read_chains('./tests/example_data/pc')

    D_KL = pc.D_KL()
    assert isinstance(D_KL, float)

    nsamples = 10
    beta = [0., 0.5, 1.]

    D_KL = pc.D_KL(nsamples=nsamples)
    assert isinstance(D_KL, WeightedLabelledSeries)
    assert D_KL.index.name == 'samples'
    assert D_KL.name == 'D_KL'
    assert_array_equal(D_KL.index, range(nsamples))

    D_KL = pc.D_KL(beta=beta)
    assert isinstance(D_KL, WeightedLabelledSeries)
    assert D_KL.index.name == 'beta'
    assert D_KL.name == 'D_KL'
    assert len(D_KL) == len(beta)

    D_KL = pc.D_KL(nsamples=nsamples, beta=beta)
    assert isinstance(D_KL, WeightedLabelledSeries)
    assert D_KL.index.names == ['beta', 'samples']
    assert D_KL.name == 'D_KL'
    assert D_KL.index.levshape == (len(beta), nsamples)

    n = 1000
    D_KL = pc.D_KL(n)

    assert abs(D_KL.mean() - pc.D_KL()) < D_KL.std() * 3


def test_d_G():
    np.random.seed(3)
    pc = read_chains('./tests/example_data/pc')

    d_G = pc.d_G()
    assert isinstance(d_G, float)

    nsamples = 10
    beta = [0., 0.5, 1.]

    d_G = pc.d_G(nsamples=nsamples)
    assert isinstance(d_G, WeightedLabelledSeries)
    assert d_G.index.name == 'samples'
    assert d_G.name == 'd_G'
    assert_array_equal(d_G.index, range(nsamples))

    d_G = pc.d_G(beta=beta)
    assert isinstance(d_G, WeightedLabelledSeries)
    assert d_G.index.name == 'beta'
    assert d_G.name == 'd_G'
    assert len(d_G) == len(beta)

    d_G = pc.d_G(nsamples=nsamples, beta=beta)
    assert isinstance(d_G, WeightedLabelledSeries)
    assert d_G.index.names == ['beta', 'samples']
    assert d_G.name == 'd_G'
    assert d_G.index.levshape == (len(beta), nsamples)

    n = 1000
    d_G = pc.d_G(n)

    assert abs(d_G.mean() - pc.d_G()) < d_G.std() * 3


def test_logL_P():
    np.random.seed(3)
    pc = read_chains('./tests/example_data/pc')

    logL_P = pc.logL_P()
    assert isinstance(logL_P, float)

    nsamples = 10
    beta = [0., 0.5, 1.]

    logL_P = pc.logL_P(nsamples=nsamples)
    assert isinstance(logL_P, WeightedLabelledSeries)
    assert logL_P.index.name == 'samples'
    assert logL_P.name == 'logL_P'
    assert_array_equal(logL_P.index, range(nsamples))

    logL_P = pc.logL_P(beta=beta)
    assert isinstance(logL_P, WeightedLabelledSeries)
    assert logL_P.index.name == 'beta'
    assert logL_P.name == 'logL_P'
    assert len(logL_P) == len(beta)

    logL_P = pc.logL_P(nsamples=nsamples, beta=beta)
    assert isinstance(logL_P, WeightedLabelledSeries)
    assert logL_P.index.names == ['beta', 'samples']
    assert logL_P.name == 'logL_P'
    assert logL_P.index.levshape == (len(beta), nsamples)

    n = 1000
    logL_P = pc.logL_P(n)

    assert abs(logL_P.mean() - pc.logL_P()) < logL_P.std() * 3


@pytest.mark.parametrize('beta', [None, 0.5, [0, 0.5, 1]])
@pytest.mark.parametrize('nsamples', [None, 10, 100])
def test_Occams_razor(nsamples, beta):
    np.random.seed(3)
    pc = read_chains('./tests/example_data/pc')
    logw = pc.logw(nsamples, beta)
    assert_allclose(pc.logZ(logw), pc.logL_P(logw) - pc.D_KL(logw))


def test_stats():
    np.random.seed(3)
    pc = read_chains('./tests/example_data/pc')

    nsamples = 10
    beta = [0., 0.5, 1.]

    vals = ['logZ', 'D_KL', 'logL_P', 'd_G']

    labels = [r'$\ln\mathcal{Z}$',
              r'$\mathcal{D}_\mathrm{KL}$',
              r'$\langle\ln\mathcal{L}\rangle_\mathcal{P}$',
              r'$d_\mathrm{G}$']

    stats = pc.stats()
    assert isinstance(stats, WeightedLabelledSeries)
    assert_array_equal(stats.drop_labels().index, vals)
    assert_array_equal(stats.get_labels(), labels)

    stats = pc.stats(nsamples=nsamples)
    assert isinstance(stats, WeightedLabelledDataFrame)
    assert_array_equal(stats.drop_labels().columns, vals)
    assert_array_equal(stats.get_labels(), labels)
    assert stats.index.name == 'samples'
    assert_array_equal(stats.index, range(nsamples))

    stats = pc.stats(beta=beta)
    assert isinstance(stats, WeightedLabelledDataFrame)
    assert_array_equal(stats.drop_labels().columns, vals)
    assert_array_equal(stats.get_labels(), labels)
    assert stats.index.name == 'beta'
    assert_array_equal(stats.index, beta)

    stats = pc.stats(nsamples=nsamples, beta=beta)
    assert isinstance(stats, WeightedLabelledDataFrame)
    assert_array_equal(stats.drop_labels().columns, vals)
    assert_array_equal(stats.get_labels(), labels)
    assert stats.index.names == ['beta', 'samples']
    assert stats.index.levshape == (len(beta), nsamples)

    for beta in [1., 0., 0.5]:
        pc.beta = beta
        n = 1000
        PC = pc.stats(n, beta)
        assert abs(pc.logZ() - PC['logZ'].mean()) < PC['logZ'].std()
        assert PC['d_G'].mean() < 5 + 3 * PC['d_G'].std()
        assert PC.cov()['D_KL']['logZ'] < 0
        assert abs(PC.logZ.mean() - pc.logZ()) < PC.logZ.std() * 3
        assert abs(PC.D_KL.mean() - pc.D_KL()) < PC.D_KL.std() * 3
        assert abs(PC.d_G.mean() - pc.d_G()) < PC.d_G.std() * 3
        assert abs(PC.logL_P.mean() - pc.logL_P()) < PC.logL_P.std() * 3

        n = 100
        assert ks_2samp(pc.logZ(n, beta), PC.logZ).pvalue > 0.05
        assert ks_2samp(pc.D_KL(n, beta), PC.D_KL).pvalue > 0.05
        assert ks_2samp(pc.d_G(n, beta), PC.d_G).pvalue > 0.05
        if beta != 0:
            assert ks_2samp(pc.logL_P(n, beta), PC.logL_P).pvalue > 0.05

    assert abs(pc.set_beta(0.0).logZ()) < 1e-2
    assert pc.set_beta(0.9).logZ() < pc.set_beta(1.0).logZ()

    assert_array_almost_equal(pc.set_beta(1).get_weights(),
                              pc.set_beta(1).get_weights())
    assert_array_almost_equal(pc.set_beta(.5).get_weights(),
                              pc.set_beta(.5).get_weights())
    assert_array_equal(pc.set_beta(0).get_weights(),
                       pc.set_beta(0).get_weights())


@pytest.mark.parametrize('kind', ['kde', 'hist', 'kde_1d', 'hist_1d',
                                  skipif_no_fastkde('fastkde_1d')])
def test_masking_1d(kind):
    pc = read_chains("./tests/example_data/pc")
    mask = pc['x0'].to_numpy() > 0
    with pytest.warns(UserWarning) if kind in ['kde',
                                               'hist'] else nullcontext():
        pc[mask].plot_1d(['x0', 'x1', 'x2'], kind=kind)


@pytest.mark.parametrize('kind', ['kde', 'scatter', 'scatter_2d', 'kde_2d',
                                  'hist_2d', skipif_no_fastkde('fastkde_2d')])
def test_masking_2d(kind):
    pc = read_chains("./tests/example_data/pc")
    mask = pc['x0'].to_numpy() > 0
    with pytest.warns(UserWarning) if kind == 'kde' else nullcontext():
        pc[mask].plot_2d(['x0', 'x1', 'x2'], kind={'lower': kind})


def test_merging():
    np.random.seed(3)
    samples_1 = read_chains('./tests/example_data/pc')
    samples_2 = read_chains('./tests/example_data/pc_250')
    samples = merge_nested_samples([samples_1, samples_2])
    nlive_1 = samples_1.nlive.mode().to_numpy()[0]
    nlive_2 = samples_2.nlive.mode().to_numpy()[0]
    nlive = samples.nlive.mode().to_numpy()[0]
    assert nlive_1 == 125
    assert nlive_2 == 250
    assert nlive == nlive_1 + nlive_2
    assert (samples_1.logZ() > samples.logZ() > samples_2.logZ()
            or samples_1.logZ() < samples.logZ() < samples_2.logZ())


def test_weighted_merging():
    # Generate some data to try it out:
    samples_1 = read_chains('./tests/example_data/pc')
    samples_2 = read_chains('./tests/example_data/pc_250')
    samples_1[('xtest', '$x_t$')] = 7*samples_1['x3']
    samples_2[('xtest', "$x_t$")] = samples_2['x3']
    mean1 = samples_1.xtest.mean()
    mean2 = samples_2.xtest.mean()

    # Test with evidence weights
    weight1 = np.exp(samples_1.logZ())
    weight2 = np.exp(samples_2.logZ())
    samples = merge_samples_weighted([samples_1, samples_2],
                                     label='Merged label')
    mean = samples.xtest.mean()
    assert np.isclose(mean, (mean1*weight1+mean2*weight2)/(weight1+weight2))

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
    mean = samples.xtest.mean()
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
    pc = read_chains("./tests/example_data/pc")
    weights = pc.get_weights()
    assert_array_equal(weights, pc.get_weights())
    assert_array_equal(pc.index.get_level_values('weights'), pc.get_weights())
    assert pc.beta == 1

    prior = pc.set_beta(0)
    assert prior.beta == 0
    assert_array_equal(prior.index.get_level_values('weights'),
                       prior.get_weights())
    assert pc.beta == 1
    assert_array_equal(pc.index.get_level_values('weights'), pc.get_weights())
    assert_array_almost_equal(sorted(prior.get_weights(), reverse=True),
                              prior.get_weights())

    for beta in np.linspace(0, 2, 10):
        pc.set_beta(beta, inplace=True)
        assert pc.beta == beta
        assert_array_equal(pc.index.get_level_values('weights'),
                           pc.get_weights())
        assert not np.array_equal(pc.index.get_level_values('weights'),
                                  weights)

    for beta in np.linspace(0, 2, 10):
        pc.beta = beta
        assert pc.beta == beta
        assert_array_equal(pc.index.get_level_values('weights'),
                           pc.get_weights())
        assert not np.array_equal(pc.index.get_level_values('weights'),
                                  weights)


def test_beta_with_logL_infinities():
    ns = read_chains("./tests/example_data/pc")
    ns.loc[:10, ('logL', r'$\ln\mathcal{L}$')] = -np.inf
    with pytest.warns(RuntimeWarning):
        ns.recompute(inplace=True)
    assert (ns.logL == -np.inf).sum() == 0


def test_prior():
    ns = read_chains("./tests/example_data/pc")
    prior = ns.prior()
    assert prior.beta == 0
    assert_frame_equal(prior, ns.set_beta(0))


def test_live_points():
    np.random.seed(4)
    pc = read_chains("./tests/example_data/pc")

    for i, logL in pc.logL.iloc[::49].items():
        live_points = pc.live_points(logL)
        assert len(live_points) == int(pc.nlive[i[0]])

        live_points_from_int = pc.live_points(i[0])
        assert_array_equal(live_points_from_int, live_points)

        live_points_from_index = pc.live_points(i)
        assert_array_equal(live_points_from_index, live_points)

    assert pc.live_points(0).index[0] == 0

    last_live_points = pc.live_points()
    logL = pc.logL_birth.max()
    assert (last_live_points.logL >= logL).all()
    assert len(last_live_points) == pc.nlive.mode().to_numpy()[0]

    assert not live_points.isweighted()


def test_dead_points():
    np.random.seed(4)
    pc = read_chains("./tests/example_data/pc")

    for i, logL in pc.logL.iloc[::49].items():
        dead_points = pc.dead_points(logL)
        assert len(dead_points) == int(len(pc[:i[0]]))

        dead_points_from_int = pc.dead_points(i[0])
        assert_array_equal(dead_points_from_int, dead_points)

        dead_points_from_index = pc.dead_points(i)
        assert_array_equal(dead_points_from_index, dead_points)

    assert pc.dead_points(1).index[0] == 0

    last_dead_points = pc.dead_points()
    logL = pc.logL_birth.max()
    assert (last_dead_points.logL <= logL).all()
    assert len(last_dead_points) == len(pc) - pc.nlive.mode().to_numpy()[0]
    assert not dead_points.isweighted()


def test_contour():
    np.random.seed(4)
    pc = read_chains("./tests/example_data/pc")

    cut_float = 30.0
    assert cut_float == pc.contour(cut_float)

    cut_int = 0
    assert pc.logL.min() == pc.contour(cut_int)

    cut_none = None
    nlive = pc.nlive.mode().to_numpy()[0]
    assert sorted(pc.logL)[-nlive] == pc.contour(cut_none)


@pytest.mark.parametrize("cut", [200, 0.0, None])
def test_truncate(cut):
    np.random.seed(4)
    pc = read_chains("./tests/example_data/pc")
    truncated_run = pc.truncate(cut)
    assert not truncated_run.index.duplicated().any()
    if cut is None:
        assert_array_equal(pc, truncated_run)


def test_hist_range_1d():
    """Test to provide a solution to #89"""
    np.random.seed(3)
    ns = read_chains('./tests/example_data/pc')
    ax = ns.plot_1d('x0', kind='hist_1d')
    x1, x2 = ax['x0'].get_xlim()
    assert x1 > -1
    assert x2 < +1
    ax = ns.plot_1d('x0', kind='hist_1d', bins=np.linspace(-1, 1, 11))
    x1, x2 = ax['x0'].get_xlim()
    assert x1 <= -1
    assert x2 >= +1


def test_contour_plot_2d_nan():
    """Contour plots with nans arising from issue #96"""
    np.random.seed(3)
    ns = read_chains('./tests/example_data/pc')

    ns.loc[:9, ('x0', '$x_0$')] = np.nan
    with pytest.raises((np.linalg.LinAlgError, RuntimeError, ValueError)):
        ns.plot_2d(['x0', 'x1'])

    # Check this error is removed in the case of zero weights
    weights = ns.get_weights()
    weights[:10] = 0
    ns.set_weights(weights, inplace=True)
    ns.plot_2d(['x0', 'x1'])


def test_compute_insertion():
    np.random.seed(3)
    ns = read_chains('./tests/example_data/pc')
    assert 'insertion' not in ns
    ns._compute_insertion_indexes()
    assert 'insertion' in ns

    nlive = ns.nlive.mode().to_numpy()[0]
    assert_array_less(ns.insertion, nlive)

    u = ns.insertion.to_numpy()/nlive
    assert kstest(u[nlive:-nlive], 'uniform').pvalue > 0.05

    pvalues = [kstest(u[i:i+nlive], 'uniform').pvalue
               for i in range(nlive, len(ns)-2*nlive, nlive)]

    assert kstest(pvalues, 'uniform').pvalue > 0.05


def test_posterior_points():
    np.random.seed(3)
    ns = read_chains('./tests/example_data/pc')
    assert_array_equal(ns.posterior_points(), ns.posterior_points())
    assert_array_equal(ns.posterior_points(0.5), ns.posterior_points(0.5))


def test_prior_points():
    ns = read_chains('./tests/example_data/pc')
    assert_array_equal(ns.prior_points(), ns.posterior_points(0))


def test_NestedSamples_importance_sample():
    np.random.seed(3)
    ns0 = read_chains('./tests/example_data/pc')
    pi0 = ns0.set_beta(0)
    NS0 = ns0.stats(nsamples=2000)

    with pytest.raises(NotImplementedError):
        ns0.importance_sample(ns0.logL, action='spam')

    ns_masked = ns0.importance_sample(ns0.logL, action='replace')
    assert_array_equal(ns0.logL, ns_masked.logL)
    assert_array_equal(ns0.logL_birth, ns_masked.logL_birth)
    assert_array_equal(ns0.get_weights(), ns_masked.get_weights())

    ns_masked = ns0.importance_sample(np.zeros_like(ns0.logL), action='add')
    assert_array_equal(ns0.logL, ns_masked.logL)
    assert_array_equal(ns0.logL_birth, ns_masked.logL_birth)
    assert_array_equal(ns0.get_weights(), ns_masked.get_weights())

    mask = ((ns0.x0 > -0.3) & (ns0.x2 > 0.2) & (ns0.x4 < 3.5)).to_numpy()
    ns_masked = merge_nested_samples((ns0[mask], ))
    V_prior = pi0[mask].get_weights().sum() / pi0.get_weights().sum()
    V_posterior = ns0[mask].get_weights().sum() / ns0.get_weights().sum()

    ns1 = ns0.importance_sample(mask, action='mask')
    assert_array_equal(ns_masked.logL, ns1.logL)
    assert_array_equal(ns_masked.logL_birth, ns1.logL_birth)
    assert_array_equal(ns_masked.get_weights(), ns1.get_weights())

    logL_new = np.where(mask, 0, -np.inf)
    ns1 = ns0.importance_sample(logL_new)
    NS1 = ns1.stats(nsamples=2000)
    assert_array_equal(ns1, ns_masked)
    logZ_V = NS0.logZ.mean() + np.log(V_posterior) - np.log(V_prior)
    assert abs(NS1.logZ.mean() - logZ_V) < 1.5 * NS1.logZ.std()

    logL_new = np.where(mask, 0, -1e30)
    ns1 = ns0.importance_sample(logL_new)
    NS1 = ns1.stats(nsamples=2000)
    logZ_V = NS0.logZ.mean() + np.log(V_posterior)
    assert abs(NS1.logZ.mean() - logZ_V) < 1.5 * NS1.logZ.std()

    ns0.importance_sample(logL_new, inplace=True)
    assert type(ns0) is NestedSamples
    assert_array_equal(ns0, ns1)
    assert ns0.root == ns1.root
    assert ns0.label == ns1.label
    assert ns0.beta == ns1.beta
    assert ns0 is not ns1


def test_MCMCSamples_importance_sample():
    np.random.seed(3)
    mc0 = read_chains('./tests/example_data/gd')

    with pytest.raises(NotImplementedError):
        mc0.importance_sample(mc0.logL, action='spam')

    # new gaussian logL
    logL_i = norm.logpdf(mc0.x3, loc=0.4, scale=0.1)

    # add logL
    mc1 = mc0.importance_sample(np.zeros_like(mc0.logL), action='add')
    assert_array_equal(mc0.logL, mc1.logL)
    assert_array_equal(mc0.get_weights(), mc1.get_weights())
    mc1 = mc0.importance_sample(logL_new=logL_i)
    assert np.all(mc1.logL.to_numpy() != mc0.logL.to_numpy())
    assert not np.all(mc1.get_weights() == mc0.get_weights())

    # replace logL
    mc2 = mc0.importance_sample(mc0.logL, action='replace')
    assert_array_equal(mc0.logL, mc2.logL)
    assert_array_equal(mc0.get_weights(), mc2.get_weights())
    mc2 = mc0.importance_sample(mc0.logL.to_numpy()+logL_i, action='replace')
    assert np.all(mc2.logL.to_numpy() != mc0.logL.to_numpy())
    assert not np.all(mc2.get_weights() == mc0.get_weights())
    assert_array_equal(mc1.logL.to_numpy(), mc2.logL.to_numpy())
    assert_array_almost_equal(mc1.logL.to_numpy(), mc2.logL.to_numpy())

    # mask logL
    mask = ((mc0.x0 > -0.3) & (mc0.x2 > 0.2) & (mc0.x4 < 3.5)).to_numpy()
    mc_masked = mc0[mask]
    mc3 = mc0.importance_sample(mask, action='mask')
    assert_array_equal(mc_masked.logL, mc3.logL)
    assert_array_equal(mc_masked.get_weights(), mc3.get_weights())
    assert np.all(mc3.x0 > -0.3)

    for mc in [mc1, mc2, mc3]:
        assert mc.root == mc0.root
        assert mc.label == mc0.label
        assert mc._metadata == mc0._metadata
        assert mc is not mc0

    mc0.importance_sample(mask, action='mask', inplace=True)
    assert isinstance(mc0, MCMCSamples)
    assert_array_equal(mc3, mc0)
    assert mc3.root == mc0.root
    assert mc3.label == mc0.label
    assert mc3._metadata == mc0._metadata
    assert mc3 is not mc0


def test_logzero_mask_prior_level():
    np.random.seed(3)
    ns0 = read_chains('./tests/example_data/pc')
    pi0 = ns0.set_beta(0)
    NS0 = ns0.stats(nsamples=2000)
    mask = ((ns0.x0 > -0.3) & (ns0.x2 > 0.2) & (ns0.x4 < 3.5)).to_numpy()

    V_prior = pi0[mask].get_weights().sum() / pi0.get_weights().sum()
    V_posterior = ns0[mask].get_weights().sum() / ns0.get_weights().sum()
    logZ_V = NS0.logZ.mean() + np.log(V_posterior) - np.log(V_prior)

    ns1 = merge_nested_samples((ns0[mask],))
    NS1 = ns1.stats(nsamples=2000)

    assert abs(NS1.logZ.mean() - logZ_V) < 1.5 * NS1.logZ.std()


def test_logzero_mask_likelihood_level():
    np.random.seed(3)
    ns0 = read_chains('./tests/example_data/pc')
    NS0 = ns0.stats(nsamples=2000)
    mask = ((ns0.x0 > -0.3) & (ns0.x2 > 0.2) & (ns0.x4 < 3.5)).to_numpy()

    V_posterior = ns0[mask].get_weights().sum() / ns0.get_weights().sum()
    logZ_V = NS0.logZ.mean() + np.log(V_posterior)

    ns1 = read_chains('./tests/example_data/pc')
    ns1.logL = np.where(mask, ns1.logL, -1e30)

    mask = ns1.logL.to_numpy() > ns1.logL_birth.to_numpy()
    ns1 = merge_nested_samples((ns1[mask],))
    NS1 = ns1.stats(nsamples=2000)

    assert abs(NS1.logZ.mean() - logZ_V) < 1.5 * NS1.logZ.std()


def test_recompute():
    np.random.seed(3)
    pc = read_chains('./tests/example_data/pc')
    recompute = pc.recompute()
    assert recompute is not pc

    pc.loc[1000, ('logL', r'$\ln\mathcal{L}$')] = pc.logL_birth.iloc[1000]-1
    with pytest.warns(RuntimeWarning):
        recompute = pc.recompute()
    assert len(recompute) == len(pc) - 1

    mn = read_chains('./tests/example_data/mn_old')
    with pytest.raises(RuntimeError):
        mn.recompute()


def test_NaN():
    np.random.seed(3)
    pc = read_chains('./tests/example_data/pc')
    with pytest.warns(RuntimeWarning, match="NaN encountered in logL."):
        pc_new = pc.copy()
        pc_new.loc[2, ('logL', r'$\ln\mathcal{L}$')] = np.nan
        pc_new.recompute(inplace=True)
    assert len(pc_new) == len(pc) - 1
    assert pc_new.nlive.iloc[0] == 124


def test_unsorted():
    np.random.seed(4)
    pc = read_chains('./tests/example_data/pc')
    i = np.random.choice(len(pc), len(pc), replace=False)
    pc_resort = NestedSamples(data=pc.loc[i, ['x0', 'x1', 'x2', 'x3', 'x4']],
                              logL=pc.loc[i, 'logL'],
                              logL_birth=pc.loc[i, 'logL_birth'])
    assert_array_equal(pc_resort, pc)


def test_copy():
    np.random.seed(3)
    pc = read_chains('./tests/example_data/pc')
    new = pc.copy()
    assert new is not pc


def test_plotting_with_integer_names():
    np.random.seed(3)
    samples_1 = Samples(data=np.random.rand(1000, 3))
    samples_2 = Samples(data=np.random.rand(1000, 3))
    samples_1.compress()
    ax = samples_1.plot_2d([0, 1, 2])
    samples_2.plot_2d(ax)

    ax = samples_1.plot_1d([0, 1, 2])
    samples_2.plot_1d(ax)

    assert samples_1[0].shape == (1000,)
    assert_array_equal(samples_1.loc[:, 0], samples_1[0])
    assert_array_equal(samples_1.loc[:, 0], samples_1.iloc[:, 0])
    with pytest.raises(KeyError):
        samples_1['0']


def test_logL_list():
    np.random.seed(5)
    default = read_chains('./tests/example_data/pc')
    logL = default.logL.tolist()
    logL_birth = default.logL_birth.tolist()
    data = default.iloc[:, :5].to_numpy().tolist()

    samples = NestedSamples(data=data, logL=logL, logL_birth=logL_birth)
    assert_array_equal(default, samples)


def test_samples_dot_plot():
    samples = read_chains('./tests/example_data/pc')
    axes = samples[['x0', 'x1', 'x2', 'x3', 'x4']].plot.hist()
    assert len(axes.containers) == 5
    fig, ax = plt.subplots()
    axes = samples.x0.plot.kde(subplots=True, ax=ax)
    assert len(axes) == 1
    axes = samples[['x0', 'x1']].plot.kde(subplots=True)
    assert len(axes) == 2

    axes = samples.plot.kde_2d('x0', 'x1')
    assert len(axes.collections) > 0
    assert axes.get_xlabel() == '$x_0$'
    assert axes.get_ylabel() == '$x_1$'
    axes = samples.plot.hist_2d('x1', 'x0')
    assert len(axes.collections) == 1
    assert axes.get_xlabel() == '$x_1$'
    assert axes.get_ylabel() == '$x_0$'
    axes = samples.plot.scatter_2d('x2', 'x3')
    assert axes.get_xlabel() == '$x_2$'
    assert axes.get_ylabel() == '$x_3$'
    assert len(axes.lines) == 1
    fig, ax = plt.subplots()
    axes = samples.x1.plot.kde_1d(ax=ax)
    assert len(axes.lines) == 1
    fig, ax = plt.subplots()
    axes = samples.x2.plot.hist_1d(ax=ax)
    assert len(axes.containers) == 1

    fig, ax = plt.subplots()
    axes = samples.x2.plot.hist_1d(ax=ax, range=[0, 0.2])
    assert axes.get_xlim()[1] < 0.3

    axes = samples.drop_labels().plot.kde_2d('x0', 'x1')
    assert len(axes.collections) > 0
    assert axes.get_xlabel() == 'x0'
    assert axes.get_ylabel() == 'x1'
    axes = samples.drop_labels().plot.hist_2d('x1', 'x0')
    assert len(axes.collections) == 1
    assert axes.get_xlabel() == 'x1'
    assert axes.get_ylabel() == 'x0'
    axes = samples.drop_labels().plot.scatter_2d('x2', 'x3')
    assert axes.get_xlabel() == 'x2'
    assert axes.get_ylabel() == 'x3'

    try:
        axes = samples.plot.fastkde_2d('x0', 'x1')
        assert axes.get_xlabel() == '$x_0$'
        assert axes.get_ylabel() == '$x_1$'
        assert len(axes.collections) > 0
        plt.close("all")
        axes = samples.drop_labels().plot.fastkde_2d('x0', 'x1')
        assert axes.get_xlabel() == 'x0'
        assert axes.get_ylabel() == 'x1'
        assert len(axes.collections) > 0
        plt.close("all")
        axes = samples.x0.plot.fastkde_1d()
        assert len(axes.lines) == 1
        plt.close("all")
        axes = samples[['x0', 'x1', 'x2', 'x3', 'x4']].plot.fastkde_1d()
        assert len(axes.lines) == 5
        plt.close("all")
    except ImportError:
        pass


@pytest.mark.parametrize('kind', ['kde', 'hist', 'kde_1d', 'hist_1d',
                                  skipif_no_fastkde('fastkde_1d')])
def test_samples_dot_plot_legend(kind):
    samples = read_chains('./tests/example_data/pc')
    fig, ax = plt.subplots()
    getattr(samples.x0.plot, kind)(ax=ax)
    getattr(samples.x1.plot, kind)(ax=ax)
    getattr(samples.x2.plot, kind)(ax=ax)
    ax.legend()
    assert ax.get_legend().get_texts()[0].get_text() == '$x_0$'
    assert ax.get_legend().get_texts()[1].get_text() == '$x_1$'
    assert ax.get_legend().get_texts()[2].get_text() == '$x_2$'


def test_fixed_width():
    samples = read_chains('./tests/example_data/pc')
    labels = samples.get_labels()
    columns = ['A really really long column label'] + list(samples.columns[1:])
    samples.columns = columns
    assert 'A really r...' in str(samples)

    mcolumns = MultiIndex.from_arrays([columns, labels])
    samples.columns = mcolumns
    assert 'A really re...' in str(WeightedLabelledDataFrame(samples))

    mcolumns = MultiIndex.from_arrays([columns, np.random.rand(len(columns))])
    samples.columns = mcolumns
    assert 'A really re...' in str(WeightedLabelledDataFrame(samples))


def test_samples_plot_labels():
    samples = read_chains('./tests/example_data/pc')
    columns = ['x0', 'x1', 'x2', 'x3', 'x4']
    axes = samples.plot_2d(columns)
    for col, ax in zip(columns, axes.loc[:, 'x0']):
        assert samples.get_label(col, 1) == ax.get_ylabel()
    for col, ax in zip(columns, axes.loc['x4', :]):
        assert samples.get_label(col, 1) == ax.get_xlabel()

    samples = samples.drop_labels()
    axes = samples.plot_2d(columns)
    for col, ax in zip(columns, axes.loc[:, 'x0']):
        assert samples.get_label(col) == ax.get_ylabel()
    for col, ax in zip(columns, axes.loc['x4', :]):
        assert samples.get_label(col) == ax.get_xlabel()

    samples.set_label('x0', 'x0')
    axes = samples.plot_2d(columns)
    for col, ax in zip(columns, axes.loc[:, 'x0']):
        assert samples.get_label(col) == ax.get_ylabel()
    for col, ax in zip(columns, axes.loc['x4', :]):
        assert samples.get_label(col) == ax.get_xlabel()


@pytest.mark.parametrize('kind', ['kde', 'hist', skipif_no_fastkde('fastkde')])
def test_samples_empty_1d_ylabels(kind):
    samples = read_chains('./tests/example_data/pc')
    columns = ['x0', 'x1', 'x2', 'x3', 'x4']

    axes = samples.plot_1d(columns, kind=kind+'_1d')
    for col in columns:
        assert axes[col].get_ylabel() == ''

    axes = samples.plot_2d(columns, kind=kind)
    for col in columns:
        assert axes[col][col].get_ylabel() == samples.get_labels_map()[col]
        assert axes[col][col].twin.get_ylabel() == ''


def test_constructors():
    samples = read_chains('./tests/example_data/pc')

    assert isinstance(samples['x0'], WeightedLabelledSeries)
    assert isinstance(samples.loc[0], WeightedLabelledSeries)
    assert samples['x0'].islabelled()
    assert samples.loc[0].islabelled()

    assert isinstance(samples.T.loc['x0'], WeightedLabelledSeries)
    assert isinstance(samples.T[0], WeightedLabelledSeries)
    assert samples.T.loc['x0'].islabelled()
    assert samples.T[0].islabelled()

    assert isinstance(samples['x0'].to_frame(), WeightedLabelledDataFrame)


def test_old_gui():
    # with pytest.raises(TypeError): TODO reinstate for >=2.1
    with pytest.raises(ValueError):
        Samples(root='./tests/example_data/gd')
    # with pytest.raises(TypeError): TODO reinstate for >=2.1
    with pytest.raises(ValueError):
        MCMCSamples(root='./tests/example_data/gd')
    # with pytest.raises(TypeError): TODO reinstate for >=2.1
    with pytest.raises(ValueError):
        NestedSamples(root='./tests/example_data/pc')

    samples = read_chains('./tests/example_data/pc')

    for kind in ['kde', 'hist']:
        with pytest.warns(UserWarning):
            samples.plot_2d(['x0', 'x1', 'x2'], kind={'lower': kind})
        with pytest.warns(UserWarning):
            samples.plot_1d(['x0', 'x1', 'x2'], kind=kind)

    with pytest.raises(ValueError):
        samples.plot_2d(['x0', 'x1', 'x2'], types={'lower': 'kde'})

    with pytest.raises(ValueError):
        samples.plot_1d(['x0', 'x1', 'x2'], plot_type='kde')

    with pytest.raises(NotImplementedError):
        samples.tex['x0'] = '$x_0$'

    with pytest.raises(NotImplementedError):
        samples.D(1000)

    with pytest.raises(NotImplementedError):
        samples.d(1000)

    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        samples.plot(ax, 'x0')
    with pytest.raises(ValueError):
        samples.plot(ax, 'x0', 'y0')

    with pytest.raises(NotImplementedError):
        make_2d_axes(['x0', 'y0'], tex={'x0': '$x_0$', 'y0': '$y_0$'})

    with pytest.raises(NotImplementedError):
        samples.ns_output(1000)

    with pytest.raises(NotImplementedError):
        make_2d_axes(['x0', 'y0'], tex={'x0': '$x_0$', 'y0': '$y_0$'})
    with pytest.raises(NotImplementedError):
        make_1d_axes(['x0', 'y0'], tex={'x0': '$x_0$', 'y0': '$y_0$'})

    with pytest.raises(NotImplementedError):
        samples.dlogX(1000)


def test_groupby_stats():
    mcmc = read_chains('./tests/example_data/cb')
    params = ['x0', 'x1']
    chains = mcmc[params + ['chain']].groupby(('chain', '$n_\\mathrm{chain}$'))

    assert chains.mean().isweighted() is True
    assert chains.std().isweighted() is True
    assert chains.median().isweighted() is True
    assert chains.var().isweighted() is True
    assert chains.kurt().isweighted() is True
    assert chains.kurtosis().isweighted() is True
    assert chains.skew().isweighted() is True
    assert chains.sem().isweighted() is True
    assert chains.corr().isweighted() is True
    assert chains.cov().isweighted() is True
    assert chains.hist().isweighted() is True
    assert chains.corrwith(mcmc).isweighted() is True

    w1 = mcmc.loc[mcmc.chain == 1].get_weights().sum()
    w2 = mcmc.loc[mcmc.chain == 2].get_weights().sum()
    assert np.all(chains.mean().get_weights() == [w1, w2])
    assert np.all(chains.std().get_weights() == [w1, w2])
    assert np.all(chains.median().get_weights() == [w1, w2])
    assert np.all(chains.var().get_weights() == [w1, w2])
    assert np.all(chains.kurt().get_weights() == [w1, w2])
    assert np.all(chains.kurtosis().get_weights() == [w1, w2])
    assert np.all(chains.skew().get_weights() == [w1, w2])
    assert np.all(chains.sem().get_weights() == [w1, w2])
    w = [w1 for _ in range(len(params))] + [w2 for _ in range(len(params))]
    assert np.all(chains.corr().get_weights() == w)
    assert np.all(chains.cov().get_weights() == w)
    assert np.all(chains.corrwith(mcmc).get_weights() == [w1, w2])

    for chain in [1, 2]:
        mask = (mcmc.chain == chain).to_numpy()
        assert_allclose(mcmc.loc[mask, params].mean(),
                        chains.mean().loc[chain])
        assert_allclose(mcmc.loc[mask, params].std(),
                        chains.std().loc[chain])
        assert_allclose(mcmc.loc[mask, params].median(),
                        chains.median().loc[chain])
        assert_allclose(mcmc.loc[mask, params].var(),
                        chains.var().loc[chain])
        assert_allclose(mcmc.loc[mask, params].kurt(),
                        chains.kurt().loc[chain])
        assert_allclose(mcmc.loc[mask, params].kurtosis(),
                        chains.kurtosis().loc[chain])
        assert_allclose(mcmc.loc[mask, params].skew(),
                        chains.skew().loc[chain])
        assert_allclose(mcmc.loc[mask, params].sem(),
                        chains.sem().loc[chain])
        assert_allclose(mcmc.loc[mask, params].cov(),
                        chains.cov().loc[chain])
        assert_allclose(mcmc.loc[mask, params].corr(),
                        chains.corr().loc[chain])
        assert_allclose([1, 1], chains.corrwith(mcmc.loc[mask, params]
                                                ).loc[chain])

        group = chains.get_group(chain).drop(
                columns=('chain', '$n_\\mathrm{chain}$'))
        assert_allclose(mcmc.loc[mask, params].mean(), group.mean())
        assert_allclose(mcmc.loc[mask, params].std(), group.std())
        assert_allclose(mcmc.loc[mask, params].median(), group.median())
        assert_allclose(mcmc.loc[mask, params].var(), group.var())
        assert_allclose(mcmc.loc[mask, params].kurt(), group.kurt())
        assert_allclose(mcmc.loc[mask, params].kurtosis(), group.kurtosis())
        assert_allclose(mcmc.loc[mask, params].skew(), group.skew())
        assert_allclose(mcmc.loc[mask, params].sem(), group.sem())
        assert_allclose(mcmc.loc[mask, params].cov(), group.cov())
        assert_allclose(mcmc.loc[mask, params].corr(), group.corr())

    assert_allclose(mcmc[params].mean(), chains.mean().mean())

    for col in params:
        if 'chain' not in col:
            for chain in [1, 2]:
                mask = (mcmc.chain == chain).to_numpy()
                assert_allclose(mcmc.loc[mask, col].mean(),
                                chains[col].mean().loc[chain])
                assert_allclose(mcmc.loc[mask, col].std(),
                                chains[col].std().loc[chain])
                assert_allclose(mcmc.loc[mask, col].median(),
                                chains[col].median().loc[chain])
                assert_allclose(mcmc.loc[mask, col].var(),
                                chains[col].var().loc[chain])
                assert_allclose(mcmc.loc[mask, col].kurt(),
                                chains[col].kurt().loc[chain])
                assert_allclose(mcmc.loc[mask, col].kurtosis(),
                                chains[col].kurtosis().loc[chain])
                assert_allclose(mcmc.loc[mask, col].skew(),
                                chains[col].skew().loc[chain])
                assert_allclose(mcmc.loc[mask, col].sem(),
                                chains[col].sem().loc[chain])
                assert_allclose(mcmc.loc[mask, col].cov(mcmc.loc[mask, col]),
                                chains[col].cov(mcmc.loc[mask, col])
                                .loc[chain])
                assert_allclose(mcmc.loc[mask, col].corr(mcmc.loc[mask, col]),
                                chains[col].corr(mcmc.loc[mask, col])
                                .loc[chain])
                q = np.random.rand()
                assert_allclose(mcmc.loc[mask, col].quantile(q),
                                chains[col].quantile(q).loc[chain])

                group = chains[col].get_group(chain)
                assert_allclose(mcmc.loc[mask, col].mean(), group.mean())
                assert_allclose(mcmc.loc[mask, col].std(), group.std())
                assert_allclose(mcmc.loc[mask, col].median(), group.median())
                assert_allclose(mcmc.loc[mask, col].var(), group.var())
                assert_allclose(mcmc.loc[mask, col].kurt(), group.kurt())
                assert_allclose(mcmc.loc[mask, col].kurtosis(),
                                group.kurtosis())
                assert_allclose(mcmc.loc[mask, col].skew(), group.skew())
                assert_allclose(mcmc.loc[mask, col].sem(), group.sem())

                assert_allclose(mcmc.loc[mask, col].cov(mcmc.loc[mask, col]),
                                group.cov(mcmc.loc[mask, col]))
                assert_allclose(mcmc.loc[mask, col].corr(mcmc.loc[mask, col]),
                                group.corr(mcmc.loc[mask, col]))

    sample = chains.sample(5)
    assert len(sample) == 10
    assert sample.value_counts('chain')[1] == 5
    assert sample.value_counts('chain')[2] == 5

    chains = mcmc.chain.groupby(mcmc.chain)
    sample = chains.sample(5)
    assert len(sample) == 10
    assert sample.value_counts()[1] == 5
    assert sample.value_counts()[2] == 5


def test_groupby_plots():
    mcmc = read_chains('./tests/example_data/cb')
    params = ['x0', 'x1']
    chains = mcmc[params + ['chain']].groupby(('chain', '$n_\\mathrm{chain}$'))
    for param in params:
        gb_plot = chains.hist(param)
        for chain in [1, 2]:
            mcmc_axes = mcmc.loc[mcmc.chain == chain].hist(param).flatten()
            gb_axes = gb_plot[chain].values[0].flatten()

            mcmc_widths = [p.get_width() for ax in mcmc_axes
                           for p in ax.patches]
            gb_widths = [p.get_width() for ax in gb_axes for p in ax.patches]
            assert_allclose(mcmc_widths, gb_widths)

            mcmc_heights = [p.get_height() for ax in mcmc_axes
                            for p in ax.patches]
            gb_heights = [p.get_height() for ax in gb_axes for p in ax.patches]
            assert_allclose(mcmc_heights, gb_heights)
            plt.close('all')

    for param in params:
        _, gb_ax = plt.subplots()
        gb_plots = chains[param].plot.hist(ax=gb_ax)
        _, mcmc_ax = plt.subplots()
        for chain, gb_ax in zip([1, 2], gb_plots):
            mcmc_ax = mcmc.loc[mcmc.chain == chain][param].plot.hist(
                    ax=mcmc_ax)
        mcmc_widths = [p.get_width() for p in mcmc_ax.patches]
        gb_widths = [p.get_width() for p in gb_ax.patches]
        assert_allclose(mcmc_widths, gb_widths)
    plt.close('all')

    for param in params:
        _, gb_ax = plt.subplots()
        gb_plots = chains[param].plot.hist_1d(ax=gb_ax)
        _, mcmc_ax = plt.subplots()
        for chain, gb_ax in zip([1, 2], gb_plots):
            mcmc_ax = mcmc.loc[mcmc.chain == chain][param].plot.hist_1d(
                    ax=mcmc_ax)
        mcmc_widths = [p.get_width() for p in mcmc_ax.patches]
        gb_widths = [p.get_width() for p in gb_ax.patches]
        assert_allclose(mcmc_widths, gb_widths)
    plt.close('all')

    for param in params:
        _, gb_ax = plt.subplots()
        gb_plots = chains[param].plot.kde(ax=gb_ax)
        _, mcmc_ax = plt.subplots()
        for chain, gb_ax in zip([1, 2], gb_plots):
            mcmc_ax = mcmc.loc[mcmc.chain == chain][param].plot.kde(
                    ax=mcmc_ax)
        [assert_allclose(m.get_data(), g.get_data())
         for m, g in zip(mcmc_ax.get_lines(), gb_ax.get_lines())]
    plt.close('all')

    for param in params:
        _, gb_ax = plt.subplots()
        gb_plots = chains[param].plot.kde_1d(ax=gb_ax)
        _, mcmc_ax = plt.subplots()
        for chain, gb_ax in zip([1, 2], gb_plots):
            mcmc_ax = mcmc.loc[mcmc.chain == chain][param].plot.kde_1d(
                    ax=mcmc_ax)
        [assert_allclose(m.get_data(), g.get_data())
         for m, g in zip(mcmc_ax.get_lines(), gb_ax.get_lines())]
    plt.close('all')

    for chain, gb_ax in zip([1, 2], chains.plot.hist_2d(*params)):
        mcmc_ax = mcmc.loc[mcmc.chain == chain].plot.hist_2d(*params)
        mcmc_widths = [p.get_width() for p in mcmc_ax.patches]
        gb_widths = [p.get_width() for p in gb_ax.patches]
        assert_allclose(mcmc_widths, gb_widths)
        mcmc_heights = [p.get_height() for p in mcmc_ax.patches]
        gb_heights = [p.get_height() for p in gb_ax.patches]
        assert_allclose(mcmc_heights, gb_heights)
        mcmc_colors = [p.get_facecolor() for p in mcmc_ax.patches]
        gb_colors = [p.get_facecolor() for p in gb_ax.patches]
        assert_allclose(mcmc_colors, gb_colors)
    plt.close('all')

    for chain, gb_ax in zip([1, 2], chains.plot.kde_2d(*params)):
        mcmc_ax = mcmc.loc[mcmc.chain == chain].plot.kde_2d(*params)
        mcmc_verts = [p.get_verts() for p in mcmc_ax.patches]
        gb_verts = [p.get_verts() for p in gb_ax.patches]
        assert_allclose(mcmc_verts, gb_verts)
        mcmc_colors = [p.get_facecolor() for p in mcmc_ax.patches]
        gb_colors = [p.get_facecolor() for p in gb_ax.patches]
        assert_allclose(mcmc_colors, gb_colors)
    plt.close('all')

    if not fastkde_mark_skip.args[0]:
        for param in params:
            _, gb_ax = plt.subplots()
            gb_plots = chains[param].plot.fastkde_1d(ax=gb_ax)
            _, mcmc_ax = plt.subplots()
            for chain, gb_ax in zip([1, 2], gb_plots):
                mcmc_ax = mcmc.loc[mcmc.chain == chain][param].plot.fastkde_1d(
                        ax=mcmc_ax)
            [assert_allclose(m.get_data(), g.get_data())
             for m, g in zip(mcmc_ax.get_lines(), gb_ax.get_lines())]
        plt.close('all')

        for chain, gb_ax in zip([1, 2], chains.plot.fastkde_2d(*params)):
            mcmc_ax = mcmc.loc[mcmc.chain == chain].plot.fastkde_2d(*params)
            mcmc_verts = [p.get_verts() for p in mcmc_ax.patches]
            gb_verts = [p.get_verts() for p in gb_ax.patches]
            assert_allclose(mcmc_verts, gb_verts)
            mcmc_colors = [p.get_facecolor() for p in mcmc_ax.patches]
            gb_colors = [p.get_facecolor() for p in gb_ax.patches]
            assert_allclose(mcmc_colors, gb_colors)
        plt.close('all')


def test_hist_1d_no_Frequency():
    np.random.seed(42)
    pc = read_chains("./tests/example_data/pc")
    axes = pc.plot_2d(['x0', 'x1', 'x2'], kind={'diagonal': 'hist_1d'})
    for i in range(len(axes)):
        assert axes.iloc[i, i].twin.get_ylabel() != 'Frequency'

    axes = pc.plot_1d(['x0', 'x1', 'x2'], kind='hist_1d')
    for ax in axes:
        assert ax.get_ylabel() != 'Frequency'

    fig, ax = plt.subplots()
    ax = pc['x0'].plot(kind='hist_1d', ax=ax)
    assert ax.get_ylabel() != 'Frequency'

    fig, ax = plt.subplots()
    ax = pc.x0.plot.hist_1d(ax=ax)
    assert ax.get_ylabel() != 'Frequency'


@pytest.mark.parametrize('kind', ['kde', 'hist'])
def test_axes_limits_1d(kind):
    np.random.seed(42)
    pc = read_chains("./tests/example_data/pc")

    axes = pc.plot_1d('x0', kind=f'{kind}_1d')
    xmin, xmax = axes['x0'].get_xlim()
    assert -0.9 < xmin < 0
    assert 0 < xmax < 0.9

    pc.x0 += 3
    pc.plot_1d(axes, kind=f'{kind}_1d')
    xmin, xmax = axes['x0'].get_xlim()
    assert -0.9 < xmin < 0
    assert 3 < xmax < 3.9

    pc.x0 -= 6
    pc.plot_1d(axes, kind=f'{kind}_1d')
    xmin, xmax = axes['x0'].get_xlim()
    assert -3.9 < xmin < -3
    assert 3 < xmax < 3.9


@pytest.mark.parametrize('kind, kwargs',
                         [('kde', {}),
                          ('hist', {'levels': [0.95, 0.68]}),
                          ])
def test_axes_limits_2d(kind, kwargs):
    np.random.seed(42)
    pc = read_chains("./tests/example_data/pc")

    axes = pc.plot_2d(['x0', 'x1'], kind=f'{kind}_2d', **kwargs)
    xmin, xmax = axes['x0']['x1'].get_xlim()
    ymin, ymax = axes['x0']['x1'].get_ylim()
    assert -0.9 < xmin < 0
    assert 0 < xmax < 0.9
    assert -0.9 < ymin < 0
    assert 0 < ymax < 0.9

    pc.x0 += 3
    pc.x1 -= 3
    pc.plot_2d(axes, kind=f'{kind}_2d', **kwargs)
    xmin, xmax = axes['x0']['x1'].get_xlim()
    ymin, ymax = axes['x0']['x1'].get_ylim()
    assert -0.9 < xmin < 0
    assert 3 < xmax < 3.9
    assert -3.9 < ymin < -3
    assert 0 < ymax < 0.9

    pc.x0 -= 6
    pc.x1 += 6
    pc.plot_2d(axes, kind=f'{kind}_2d', **kwargs)
    xmin, xmax = axes['x0']['x1'].get_xlim()
    ymin, ymax = axes['x0']['x1'].get_ylim()
    assert -3.9 < xmin < -3
    assert 3 < xmax < 3.9
    assert -3.9 < ymin < -3
    assert 3 < ymax < 3.9
