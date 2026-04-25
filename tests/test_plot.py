import anesthetic.examples._matplotlib_agg  # noqa: F401
from packaging import version
from warnings import catch_warnings, filterwarnings
import pytest
from pytest import approx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from anesthetic.plot import (make_1d_axes, make_2d_axes, kde_plot_1d,
                             fastkde_plot_1d, hist_plot_1d, hist_plot_2d,
                             fastkde_contour_plot_2d, kde_contour_plot_2d,
                             scatter_plot_2d, quantile_plot_interval,
                             basic_cmap, _plot_window, _basis_aligned_grid,
                             AxesSeries, AxesDataFrame)
from anesthetic.examples.perfect_ns import correlated_gaussian
from numpy.testing import assert_array_equal, assert_allclose

from matplotlib.axes import SubplotBase
from matplotlib.contour import ContourSet
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.colors import ColorConverter, to_rgba
from matplotlib.figure import Figure
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy import stats
from utils import skipif_no_fastkde, astropy_mark_xfail, fastkde_mark_xfail


@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    yield
    plt.close("all")


def test_AxesObjects():
    paramnames = ['a', 'b', 'c']

    # AxesSeries
    axes = AxesSeries(index=paramnames)
    assert isinstance(axes, AxesSeries)
    assert isinstance(axes.iloc[0], SubplotBase)
    assert axes.iloc[0].get_xlabel() == 'a'
    axes.set_xlabels(labels=dict(a='A', b='B', c='C'))
    assert axes.iloc[0].get_xlabel() == 'A'
    axes.tick_params(labelrotation=0, labelsize='medium')

    # AxesDataFrame
    axes = AxesDataFrame(index=paramnames + ['d'], columns=paramnames)
    assert isinstance(axes, AxesDataFrame)
    assert isinstance(axes.iloc[0, 0], SubplotBase)
    assert axes.iloc[-1, 0].get_xlabel() == 'a'
    assert axes.iloc[-1, 0].get_ylabel() == 'd'
    axes.set_labels(labels=dict(a='A', b='B', c='C', d='D'))
    assert axes.iloc[-1, 0].get_xlabel() == 'A'
    assert axes.iloc[-1, 0].get_ylabel() == 'D'
    axes.tick_params(labelrotation=0, labelsize='medium')
    xmin1, xmax1 = axes.iloc[0, 0].get_xlim()
    ymin1, ymax1 = axes.iloc[0, 0].get_ylim()
    axes.set_margins(m=0.5)
    xmin2, xmax2 = axes.iloc[0, 0].get_xlim()
    ymin2, ymax2 = axes.iloc[0, 0].get_ylim()
    assert xmin2 == xmin1 - 0.5 * (xmax1 - xmin1)
    assert xmax2 == xmax1 + 0.5 * (xmax1 - xmin1)
    assert ymin2 == ymin1 - 0.5 * (ymax1 - ymin1)
    assert ymax2 == ymax1 + 0.5 * (ymax1 - ymin1)


def test_make_1d_axes():
    paramnames = ['A', 'B', 'C', 'D', 'E']
    labels = {'A': 'tA', 'B': 'tB', 'C': 'tC', 'D': 'tD', 'E': 'tE'}

    # Check no optional arguments
    fig, axes = make_1d_axes(paramnames)
    assert isinstance(fig, Figure)
    assert isinstance(axes, AxesSeries)
    assert isinstance(axes.to_frame(), AxesDataFrame)
    assert_array_equal(axes.index, paramnames)
    for p, ax in axes.items():
        assert ax.get_xlabel() == p

    # Check single string input
    fig, axes = make_1d_axes(paramnames[0])
    assert isinstance(fig, Figure)
    assert isinstance(axes, AxesSeries)
    assert axes.index.size == 1
    assert_array_equal(axes.index, paramnames[0])

    # Check labels argument
    fig, axes = make_1d_axes(paramnames, labels=labels)
    for t in labels:
        assert axes[t].get_xlabel() != t
        assert axes[t].get_xlabel() == labels[t]

    # Check fig argument
    fig0 = plt.figure()
    fig0.suptitle('hi there')
    fig, axes = make_1d_axes(paramnames)
    assert fig is not fig0
    fig, axes = make_1d_axes(paramnames, fig=fig0)
    assert fig is fig0

    # Check ncol argument
    fig, axes = make_1d_axes(paramnames, ncol=2)
    nrows, ncol = axes.iloc[0].get_subplotspec().get_gridspec().get_geometry()
    assert ncol == 2

    # Check gridspec argument
    grid = gs.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
    g00 = grid[0, 0]
    fig, axes = make_1d_axes(paramnames, subplot_spec=g00)
    assert g00 is axes.iloc[0].get_subplotspec().get_topmost_subplotspec()

    # Check gridspec kwargs
    fig, axes = make_1d_axes(paramnames, gridspec_kw=dict(wspace=0.1))
    ws = axes['A'].get_subplotspec().get_gridspec().get_subplot_params().wspace
    assert ws == 0.1

    # Check figure kwargs
    fig, axes = make_1d_axes(paramnames, figsize=(5, 5))
    assert fig.get_figwidth() == 5
    assert fig.get_figheight() == 5

    # Check unexpected kwargs
    with pytest.raises((AttributeError, TypeError)):
        make_1d_axes(paramnames, spam='ham')


@pytest.mark.parametrize('params', [[0, 1, 2, 3],
                                    [0, 2, 3],
                                    [1, 2, 3],
                                    ['x0', 'x1', 'x2'],
                                    ['1', '2', '3'],
                                    [0, 'x1', 'x2'],
                                    ])
def test_make_Nd_axes_integers(params):
    fig, axes = make_1d_axes(params)
    assert isinstance(fig, Figure)
    assert isinstance(axes, AxesSeries)
    assert list(axes.index) == params

    fig, axes = make_2d_axes(params)
    assert isinstance(fig, Figure)
    assert isinstance(axes, AxesDataFrame)
    assert list(axes.index) == params
    assert list(axes.columns) == params


def test_make_2d_axes_inputs_outputs():
    paramnames_x = ['A', 'B', 'C', 'D']
    paramnames_y = ['B', 'A', 'D', 'E']

    # 2D axes
    fig, axes = make_2d_axes([paramnames_x, paramnames_y])
    assert isinstance(fig, Figure)
    assert isinstance(axes, AxesDataFrame)
    assert isinstance(axes['A'], AxesSeries)
    assert isinstance(axes.loc['A':'B', 'B':'C'], AxesDataFrame)
    assert_array_equal(axes.index, paramnames_y)
    assert_array_equal(axes.columns, paramnames_x)

    # Axes labels
    for p, ax in axes.iloc[:, 0].items():
        assert ax.get_ylabel() == p

    for p, ax in axes.iloc[-1].items():
        assert ax.get_xlabel() == p

    for ax in axes.iloc[:-1, 1:].to_numpy().flatten():
        assert ax.get_xlabel() == ''
        assert ax.get_ylabel() == ''

    # Check fig argument
    fig0 = plt.figure()
    fig, axes = make_2d_axes(paramnames_x)
    assert fig is not fig0
    fig, axes = make_2d_axes(paramnames_x, fig=fig0)
    assert fig is fig0

    # Check gridspec argument
    grid = gs.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
    g00 = grid[0, 0]
    fig, axes = make_2d_axes(paramnames_x, subplot_spec=g00)
    assert g00 is axes.iloc[0, 0].get_subplotspec().get_topmost_subplotspec()

    # Check gridspec kwargs
    fig, axes = make_1d_axes(paramnames_x, gridspec_kw=dict(wspace=0.1))
    ws = axes['A'].get_subplotspec().get_gridspec().get_subplot_params().wspace
    assert ws == 0.1

    # Check figure kwargs
    fig, axes = make_2d_axes(paramnames_x, figsize=(5, 5))
    assert fig.get_figwidth() == 5
    assert fig.get_figheight() == 5

    # Check unexpected kwargs
    with pytest.raises((AttributeError, TypeError)):
        make_2d_axes(paramnames_x, spam='ham')


@pytest.mark.parametrize('paramnames_y', [['A', 'B', 'C', 'D'],
                                          ['A', 'C', 'B', 'D'],
                                          ['D', 'C', 'B', 'A'],
                                          ['C', 'B', 'A'],
                                          ['E', 'F', 'G', 'H'],
                                          ['A', 'B', 'E', 'F'],
                                          ['B', 'E', 'A', 'F'],
                                          ['B', 'F', 'A', 'H', 'G'],
                                          ['B', 'A', 'H', 'G']])
@pytest.mark.parametrize('upper', [False, True])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('diagonal', [False, True])
def test_make_2d_axes_behaviour(diagonal, lower, upper, paramnames_y):
    np.random.seed(0)

    def calc_n(axes):
        """Compute the number of upper, lower and diagonal plots."""
        n = {'upper': 0, 'lower': 0, 'diagonal': 0}
        for y, row in axes.iterrows():
            for x, ax in row.items():
                if ax is not None:
                    n[ax.position] += 1
        return n

    # Check for only paramnames_x
    paramnames_x = ['A', 'B', 'C', 'D']
    nx = len(paramnames_x)
    fig, axes = make_2d_axes(paramnames_x,
                             upper=upper,
                             lower=lower,
                             diagonal=diagonal)
    ns = calc_n(axes)
    assert ns['upper'] == upper * nx*(nx-1)//2
    assert ns['lower'] == lower * nx*(nx-1)//2
    assert ns['diagonal'] == diagonal * nx

    params = [paramnames_x, paramnames_y]
    all_params = paramnames_x + paramnames_y
    nu, nl, nd = 0, 0, 0
    for x in paramnames_x:
        for y in paramnames_y:
            if x == y:
                nd += 1
            elif all_params.index(x) < all_params.index(y):
                nl += 1
            elif all_params.index(x) > all_params.index(y):
                nu += 1
    fig, axes = make_2d_axes(params,
                             upper=upper,
                             lower=lower,
                             diagonal=diagonal)
    ns = calc_n(axes)
    assert ns['upper'] == upper * nu
    assert ns['lower'] == lower * nl
    assert ns['diagonal'] == diagonal * nd


@pytest.mark.parametrize('upper', [False, True])
@pytest.mark.parametrize('ticks', ['inner', 'outer', None])
def test_make_2d_axes_ticks(upper, ticks):
    xticks = [0.1, 0.4, 0.7]
    yticks = [0.2, 0.5, 0.8]
    paramnames = ["x0", "x1", "x2", "x3"]
    for k in paramnames:
        fig, axes = make_2d_axes(paramnames, upper=upper, ticks=ticks)
        axes[k][k].set_xticks(xticks)
        axes[k][k].set_yticks(yticks)
        for i, row in axes.iterrows():
            for j, ax in row.items():
                if ax is None:
                    break
                if i == k:
                    assert np.array_equal(yticks, ax.get_yticks())
                else:
                    assert not np.array_equal(yticks, ax.get_yticks())
                if j == k:
                    assert np.array_equal(xticks, ax.get_xticks())
                else:
                    assert not np.array_equal(xticks, ax.get_xticks())
    with pytest.raises(ValueError):
        make_2d_axes(paramnames, upper=upper, ticks='spam')


def test_make_2d_axes_ticks_error():
    with pytest.raises(ValueError):
        make_2d_axes(['a', 'b'], ticks='spam')


def test_2d_axes_limits():
    np.random.seed(0)
    paramnames = ['A', 'B', 'C', 'D']
    fig, axes = make_2d_axes(paramnames)
    for x in paramnames:
        for y in paramnames:
            a, b, c, d = np.random.rand(4)
            axes[x][y].set_xlim(a, b)
            for z in paramnames:
                assert axes[x][z].get_xlim() == (a, b)
                assert axes[z][x].get_ylim() == (a, b)

            axes[x][y].set_ylim(c, d)
            for z in paramnames:
                assert axes[y][z].get_xlim() == (c, d)
                assert axes[z][y].get_ylim() == (c, d)


@pytest.mark.parametrize('axesparams', [['A', 'B', 'C', 'D'],
                                        [['A', 'B', 'C', 'D'], ['A', 'B']],
                                        [['A', 'B'], ['A', 'B', 'C', 'D']]])
@pytest.mark.parametrize('params', [{'A': 0},
                                    {'A': 0, 'C': 0, 'E': 0},
                                    {'A': 0, 'C': [0, 0.5]}])
@pytest.mark.parametrize('upper', [True, False])
def test_2d_axes_axlines(axesparams, params, upper):
    kwargs = dict(c='k', ls='--', lw=0.5)
    fig, axes = make_2d_axes(axesparams, upper=upper)
    axes.axlines(params, **kwargs)


@pytest.mark.parametrize('axesparams', [['A', 'B', 'C', 'D'],
                                        [['A', 'B', 'C', 'D'], ['A', 'B']],
                                        [['A', 'B'], ['A', 'B', 'C', 'D']]])
@pytest.mark.parametrize('params', [{'A': (0, 0.1)},
                                    {'A': (0, 1), 'C': (0, 1), 'E': (0, 1)},
                                    {'A': (0, 1), 'C': [(-0.1, 0), (0.5, 1)]}])
@pytest.mark.parametrize('upper', [True, False])
def test_2d_axes_axspans(axesparams, params, upper):
    kwargs = dict(c='k', alpha=0.5)
    fig, axes = make_2d_axes(axesparams, upper=upper)
    axes.axspans(params, **kwargs)


@pytest.mark.parametrize('axesparams', [['A', 'B', 'C', 'D'],
                                        [['A', 'B', 'C', 'D'], ['A', 'B']],
                                        [['A', 'B'], ['A', 'B', 'C', 'D']]])
@pytest.mark.parametrize('params', [{'A': 0},
                                    {'A': 0, 'C': 0, 'E': 0},
                                    {'A': [0, 0.1], 'C': [0, 0.5]}])
@pytest.mark.parametrize('upper', [True, False])
def test_2d_axes_scatter(axesparams, params, upper):
    kwargs = dict(c='k', marker='*')
    fig, axes = make_2d_axes(axesparams, upper=upper)
    axes.scatter(params, **kwargs)


@pytest.mark.parametrize('plot_1d', [kde_plot_1d,
                                     skipif_no_fastkde(fastkde_plot_1d)])
def test_kde_plot_1d(plot_1d):
    fig, ax = plt.subplots()
    np.random.seed(0)
    data = np.random.randn(1000)

    # Check height
    line, = plot_1d(ax, data)
    assert isinstance(line, Line2D)
    assert line.get_ydata().max() <= 1

    # Check arguments are passed onward to underlying function
    line, = plot_1d(ax, data, color='r')
    assert line.get_color() == 'r'
    line, = plot_1d(ax, data, cmap=plt.cm.Blues)
    assert line.get_color() == plt.cm.Blues(0.68)

    # Check q
    plot_1d(ax, data, q='1sigma')
    plot_1d(ax, data, q=0)
    plot_1d(ax, data, q=1)
    plot_1d(ax, data, q=5)
    plot_1d(ax, data, q=10)
    plot_1d(ax, data, q=0.1)
    plot_1d(ax, data, q=0.9)
    plot_1d(ax, data, q=(0.1, 0.9))

    # Check iso-probability code
    line, fill = plot_1d(ax, data, facecolor=True)
    plot_1d(ax, data, facecolor=True, levels=[0.8, 0.6, 0.2])
    line, fill = plot_1d(ax, data, fc='blue', color='k', ec='r')
    assert np.all(fill[0].get_edgecolor()[0] == to_rgba('r'))
    assert (to_rgba(line[0].get_color()) == to_rgba('r'))
    line, fill = plot_1d(ax, data, fc=True, color='k', ec=None)
    assert len(fill[0].get_edgecolor()) == 0
    assert (to_rgba(line[0].get_color()) == to_rgba('k'))

    # Check levels
    with pytest.raises(ValueError):
        fig, ax = plt.subplots()
        plot_1d(ax, data, fc=True, levels=[0.68, 0.95])

    # Check xlim, Gaussian (i.e. limits reduced to relevant data range)
    fig, ax = plt.subplots()
    data = np.random.randn(1000) * 0.01 + 0.5
    plot_1d(ax, data)
    xmin, xmax = ax.get_xlim()
    assert xmin > 0.4
    assert xmax < 0.6
    # Check xlim, Uniform (i.e. data and limits span entire prior boundary)
    fig, ax = plt.subplots()
    data = np.random.uniform(size=1000)
    plot_1d(ax, data, q=0)
    xmin, xmax = ax.get_xlim()
    assert xmin <= 0
    assert xmax >= 1


@fastkde_mark_xfail
def test_fastkde_min_max():
    np.random.seed(0)
    data_x = np.random.randn(100)
    data_y = np.random.randn(100)
    xmin, xmax = -1, +1
    ymin, ymax = -1, +1
    _, ax = plt.subplots()
    line, = fastkde_plot_1d(ax, data_x, xmin=xmin)
    assert (line.get_xdata() >= xmin).all()

    _, ax = plt.subplots()
    line, = fastkde_plot_1d(ax, data_x, xmax=xmax)
    assert (line.get_xdata() <= xmax).all()

    _, ax = plt.subplots()
    line, = fastkde_plot_1d(ax, data_x, xmin=xmin, xmax=xmax)
    assert (line.get_xdata() >= xmin).all()
    assert (line.get_xdata() <= xmax).all()

    _, ax = plt.subplots()
    fastkde_contour_plot_2d(ax, data_x, data_y, xmin=xmin, ymin=ymin)
    assert ax.get_xlim()[0] >= xmin
    assert ax.get_ylim()[0] >= ymin

    _, ax = plt.subplots()
    fastkde_contour_plot_2d(ax, data_x, data_y, xmax=xmax, ymax=ymax)
    assert ax.get_xlim()[1] <= xmax
    assert ax.get_ylim()[1] <= ymax

    _, ax = plt.subplots()
    fastkde_contour_plot_2d(ax, data_x, data_y,
                            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    assert ax.get_xlim()[0] >= xmin
    assert ax.get_xlim()[1] <= xmax
    assert ax.get_ylim()[0] >= ymin
    assert ax.get_ylim()[1] <= ymax


def test_hist_plot_1d():
    fig, ax = plt.subplots()
    np.random.seed(0)
    data = np.random.randn(1000)

    # Check heights for histtype 'bar'
    bars = hist_plot_1d(ax, data, histtype='bar')[2]
    assert np.all([isinstance(b, Patch) for b in bars])
    assert max([b.get_height() for b in bars]) == 1.
    assert np.all(np.array([b.get_height() for b in bars]) <= 1.)

    # Check heights for histtype 'step'
    polygon, = hist_plot_1d(ax, data, histtype='step')[2]
    assert isinstance(polygon, Polygon)
    trans = polygon.get_transform() - ax.transData
    assert np.isclose(trans.transform(polygon.xy)[:, -1].max(), 1.,
                      rtol=1e-10, atol=1e-10)
    assert np.all(trans.transform(polygon.xy)[:, -1] <= 1.)

    # Check heights for histtype 'stepfilled'
    polygon, = hist_plot_1d(ax, data, histtype='stepfilled')[2]
    assert isinstance(polygon, Polygon)
    trans = polygon.get_transform() - ax.transData
    assert np.isclose(trans.transform(polygon.xy)[:, -1].max(), 1.,
                      rtol=1e-10, atol=1e-10)
    assert np.all(trans.transform(polygon.xy)[:, -1] <= 1.)

    # Check arguments are passed onward to underlying function
    bars = hist_plot_1d(ax, data, histtype='bar', color='r', alpha=0.5)[2]
    cc = ColorConverter.to_rgba('r', alpha=0.5)
    assert np.all([b.get_fc() == cc for b in bars])
    bars = hist_plot_1d(ax, data, histtype='bar', cmap=plt.cm.viridis,
                        alpha=0.5)[2]
    cc = ColorConverter.to_rgba(plt.cm.viridis(0.68), alpha=0.5)
    assert np.all([b.get_fc() == cc for b in bars])
    polygon, = hist_plot_1d(ax, data, histtype='step', color='r', alpha=0.5)[2]
    assert polygon.get_ec() == ColorConverter.to_rgba('r', alpha=0.5)
    polygon, = hist_plot_1d(ax, data, histtype='step', cmap=plt.cm.viridis,
                            color='r')[2]
    assert polygon.get_ec() == ColorConverter.to_rgba('r')


@astropy_mark_xfail
@pytest.mark.parametrize('bins', ['knuth', 'freedman', 'blocks'])
def test_astropyhist_plot_1d(bins):
    fig, ax = plt.subplots()
    np.random.seed(0)
    data = np.random.randn(100)
    with pytest.raises(ValueError):
        hist_plot_1d(ax, data, bins=bins)


@pytest.mark.parametrize('bins', ['fd', 'scott', 'sqrt'])
def test_hist_plot_1d_bins(bins):
    np.random.seed(0)
    num = 1000
    bound = 5
    x = np.random.uniform(-bound, +bound, num)
    w = stats.norm.pdf(x)
    fig, ax = plt.subplots()
    _, edges, _ = hist_plot_1d(ax, x, weights=w, bins=bins)
    assert 15 <= edges.size <= 30
    assert edges[+0] > x.min()
    assert edges[-1] < x.max()

    _, edges1, _ = hist_plot_1d(ax, x, weights=w, bins=bins, beta=1)
    _, edges2, _ = hist_plot_1d(ax, x, weights=w, bins=bins, beta=2)
    assert edges1.size > edges2.size > edges.size

    _, edgesr, _ = hist_plot_1d(ax, x, weights=w, bins=bins, range=(-3, 3))
    assert 10 <= edgesr.size <= edges.size
    assert edgesr[0] == -3
    assert edgesr[-1] == 3

    _, edges, _ = hist_plot_1d(ax, x, weights=w, bins=bins, range=None)
    assert 15 <= edges.size <= 30
    assert edges[+0] == pytest.approx(x.min(), abs=2*bound/num)
    assert edges[-1] == pytest.approx(x.max(), abs=2*bound/num)

    _, edges, _ = hist_plot_1d(ax, x, weights=None, bins=bins)
    assert 10 <= edges.size <= 35
    assert edges[+0] == pytest.approx(x.min(), abs=2*bound/num)
    assert edges[-1] == pytest.approx(x.max(), abs=2*bound/num)


def test_hist_plot_2d():
    fig, ax = plt.subplots()
    np.random.seed(0)
    data_x, data_y = np.random.randn(2, 1000)
    hist_plot_2d(ax, data_x, data_y)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    assert xmin > -5 and xmax < 5 and ymin > -5 and ymax < 5

    fig, ax = plt.subplots()
    data_x, data_y = np.random.uniform(-10, 10, (2, 1000))
    hist_plot_2d(ax, data_x, data_y)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    assert xmin > -10 and xmax < 10 and ymin > -10 and ymax < 10

    fig, ax = plt.subplots()
    data_x, data_y = np.random.uniform(-10, 10, (2, 1000))
    weights = np.exp(-(data_x**2 + data_y**2)/2)
    hist_plot_2d(ax, data_x, data_y, weights=weights, bins=30)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    assert xmin > -6 and xmax < 6 and ymin > -6 and ymax < 6

    fig, ax = plt.subplots()
    data_x, data_y = np.random.randn(2, 1000)
    hist_plot_2d(ax, data_x, data_y, levels=[0.95, 0.68], cmin=50)
    hist_plot_2d(ax, data_x, data_y, levels=[0.95, 0.68], cmax=50)


@pytest.mark.parametrize('plot_1d', [kde_plot_1d,
                                     skipif_no_fastkde(fastkde_plot_1d)])
@pytest.mark.parametrize('s', [1, 2])
def test_1d_density_kwarg(plot_1d, s):
    np.random.seed(0)
    x = np.random.normal(scale=s, size=2000)
    fig, ax = plt.subplots()

    # hist density = False:
    h = hist_plot_1d(ax, x, density=False,
                     bins=np.linspace(-5.5, 5.5, 12))[2]
    bar_height = h.get_children()[len(h.get_children()) // 2].get_height()
    assert bar_height == pytest.approx(1, rel=0.1)

    # kde density = False:
    k = plot_1d(ax, x, density=False)[0]
    f = interp1d(k.get_xdata(), k.get_ydata(), 'cubic', assume_sorted=True)
    kde_height = f(0)
    assert kde_height == pytest.approx(1, rel=0.1)

    # hist density = True:
    h = hist_plot_1d(ax, x, density=True,
                     bins=np.linspace(-5.5, 5.5, 12))[2]
    bar_height = h.get_children()[len(h.get_children()) // 2].get_height()
    assert bar_height == pytest.approx(erf(0.5 / np.sqrt(2) / s), rel=0.1)

    # kde density = True:
    k = plot_1d(ax, x, density=True)[0]
    f = interp1d(k.get_xdata(), k.get_ydata(), 'cubic', assume_sorted=True)
    kde_height = f(0)
    gauss_norm = 1 / np.sqrt(2 * np.pi * s**2)
    assert kde_height == pytest.approx(gauss_norm, rel=0.1)


@pytest.mark.parametrize('contour_plot_2d',
                         [kde_contour_plot_2d,
                          skipif_no_fastkde(fastkde_contour_plot_2d)])
def test_contour_plot_2d(contour_plot_2d):
    fig, ax = plt.subplots()
    np.random.seed(1)
    data_x = np.random.randn(1000)
    data_y = np.random.randn(1000)
    cf, ct = contour_plot_2d(ax, data_x, data_y)
    assert isinstance(cf, ContourSet)
    assert isinstance(ct, ContourSet)

    # Check levels
    with pytest.raises(ValueError):
        fig, ax = plt.subplots()
        contour_plot_2d(ax, data_x, data_y, levels=[0.68, 0.95])

    # Check unfilled
    cmap = basic_cmap('C2')
    fig, ax = plt.subplots()
    cf1, ct1 = contour_plot_2d(ax, data_x, data_y, facecolor='C2')
    cf2, ct2 = contour_plot_2d(ax, data_x, data_y, fc='None', cmap=cmap)
    # filled `contourf` and unfilled `contour` colors are the same:
    # tcolors deprecated in matplotlib 3.8
    cf1_tcolors = [tuple(rgba) for rgba in cf1.to_rgba(cf1.cvalues, cf1.alpha)]
    ct2_tcolors = [tuple(rgba) for rgba in ct2.to_rgba(ct2.cvalues, ct2.alpha)]

    assert cf1_tcolors[0] == ct2_tcolors[0]
    assert cf1_tcolors[1] == ct2_tcolors[1]
    cf, ct = contour_plot_2d(ax, data_x, data_y, edgecolor='C0')
    assert ct.colors == 'C0'
    cf, ct = contour_plot_2d(ax, data_x, data_y, ec='C0', cmap=plt.cm.Reds)
    assert cf.get_cmap() == plt.cm.Reds
    assert ct.colors == 'C0'
    fig, ax = plt.subplots()
    cf, ct = contour_plot_2d(ax, data_x, data_y, fc=None)
    assert cf is None
    assert ct.colors is None
    assert ct.get_cmap()(1.) == to_rgba('C0')
    cf, ct = contour_plot_2d(ax, data_x, data_y, fc=None, c='C3')
    assert cf is None
    assert ct.colors is None
    assert ct.get_cmap()(1.) == to_rgba('C3')
    cf, ct = contour_plot_2d(ax, data_x, data_y, fc=None, ec='C1')
    assert cf is None
    assert ct.colors == 'C1'
    cf, ct = contour_plot_2d(ax, data_x, data_y, fc=None, cmap=plt.cm.Reds)
    assert cf is None
    assert ct.colors is None
    assert ct.get_cmap() == plt.cm.Reds

    # Check limits, Gaussian (i.e. limits reduced to relevant data range)
    fig, ax = plt.subplots()
    data_x = np.random.randn(1000) * 0.01 + 0.5
    data_y = np.random.randn(1000) * 0.01 + 0.5
    contour_plot_2d(ax, data_x, data_y)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    assert xmin > 0.4
    assert xmax < 0.6
    assert ymin > 0.4
    assert ymax < 0.6
    # Check limits, Uniform (i.e. data & limits span entire prior boundary)
    fig, ax = plt.subplots()
    data_x = np.random.uniform(size=1000)
    data_y = np.random.uniform(size=1000)
    contour_plot_2d(ax, data_x, data_y, q=0)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if contour_plot_2d is fastkde_contour_plot_2d:
        assert xmin <= 0
        assert xmax >= 1
        assert ymin <= 0
        assert ymax >= 1
    elif contour_plot_2d is kde_contour_plot_2d:
        assert xmin == pytest.approx(0, abs=0.01)
        assert xmax == pytest.approx(1, abs=0.01)
        assert ymin == pytest.approx(0, abs=0.01)
        assert ymax == pytest.approx(1, abs=0.01)


@pytest.mark.parametrize('plot_1d', [kde_plot_1d, hist_plot_1d])
def test_q_1d(plot_1d):
    np.random.seed(42)
    d = np.random.randn(1000)
    fig, ax = plt.subplots()

    # check that axis limits match data limits for single plot
    p = plot_1d(ax, d, q=1)  # int input for q
    p = p[1] if plot_1d == hist_plot_1d else p[0].get_data()[0]
    assert ax.get_xlim() == pytest.approx((-1, 1), rel=0.1)
    assert (p.min(), p.max()) == pytest.approx((-1, 1), rel=0.1)

    # check that axis limits grow with new, larger plot
    p = plot_1d(ax, d, q="2sigma")  # str input for q
    p = p[1] if plot_1d == hist_plot_1d else p[0].get_data()[0]
    assert ax.get_xlim() == pytest.approx((-2, 2), rel=0.15)
    assert (p.min(), p.max()) == pytest.approx((-2, 2), rel=0.1)

    # check that axis limits do not shrink with new, smaller plot
    p = plot_1d(ax, d, q=(0.025, 0.84))  # tuple input for q
    p = p[1] if plot_1d == hist_plot_1d else p[0].get_data()[0]
    assert ax.get_xlim() == pytest.approx((-2, 2), rel=0.15)
    assert (p.min(), p.max()) == pytest.approx((-2, 1), rel=0.1)


@pytest.mark.parametrize('plot_2d', [kde_contour_plot_2d,
                                     hist_plot_2d,
                                     scatter_plot_2d])
def test_q_2d(plot_2d):
    np.random.seed(42)
    d = np.random.randn(2, 1000)

    def get_data_from_plot(plot_2d, p):
        if plot_2d == kde_contour_plot_2d:
            # p[0].allsegs[i] is the list of polygon outlines drawn at
            # contourf level i; each outline is an (N, 2) vertex array.
            # Flatten the polygons within each level, skipping empty levels.
            level_polygons = [np.concatenate(polygons)
                              for polygons in p[0].allsegs if polygons]
            x, y = np.concatenate(level_polygons).T
        elif plot_2d == hist_plot_2d:
            x = p.get_coordinates()[0, :, 0]
            y = p.get_coordinates()[:, 0, 1]
        elif plot_2d == scatter_plot_2d:
            x, y = p[0].get_xydata().T
        return x, y

    def expected_limits(plot_2d, x, y):
        if plot_2d == scatter_plot_2d:
            xpad = (x.max()-x.min()) * plt.rcParams['axes.xmargin']
            ypad = (y.max()-y.min()) * plt.rcParams['axes.ymargin']
            return (x.min()-xpad, x.max()+xpad), (y.min()-ypad, y.max()+ypad)
        return (x.min(), x.max()), (y.min(), y.max())

    fig, ax = plt.subplots()
    # check that axis limits match data limits for single plot
    p = plot_2d(ax, d[0], d[1], q=1)  # int input for q
    x, y = get_data_from_plot(plot_2d, p)
    xlim, ylim = expected_limits(plot_2d, x, y)
    assert ax.get_xlim() == pytest.approx((-1, 1), rel=0.2)
    assert ax.get_ylim() == pytest.approx((-1, 1), rel=0.2)
    assert ax.get_xlim() == pytest.approx(xlim, rel=1e-15)
    assert ax.get_ylim() == pytest.approx(ylim, rel=1e-15)

    # check that axis limits grow with new, larger plot
    p = plot_2d(ax, d[0], d[1], q="2sigma")  # str input for q
    x, y = get_data_from_plot(plot_2d, p)
    xlim, ylim = expected_limits(plot_2d, x, y)
    assert ax.get_xlim() == pytest.approx((-2, 2), rel=0.2)
    assert ax.get_ylim() == pytest.approx((-2, 2), rel=0.2)
    assert ax.get_xlim() == pytest.approx(xlim, rel=1e-15)
    assert ax.get_ylim() == pytest.approx(ylim, rel=1e-15)

    # check that axis limits do not shrink with new, smaller plot
    p = plot_2d(ax, d[0], d[1], q=(0.025, 0.84))  # tuple input for q
    x, y = get_data_from_plot(plot_2d, p)
    assert ax.get_xlim() == pytest.approx((-2, 2), rel=0.2)
    assert ax.get_ylim() == pytest.approx((-2, 2), rel=0.2)
    assert (x.min(), x.max()) == pytest.approx((-2, 1), rel=0.2)
    assert (y.min(), y.max()) == pytest.approx((-2, 1), rel=0.2)


def test_q_invariant_levels():
    """Contour levels must not depend on the plotting window set by `q`.

    Regression test for a bug where ``iso_probability_contours`` was
    computed from the KDE on the plotting meshgrid, so narrowing ``q``
    re-ranked the visible mass and shifted the contours (see PR #431).
    """
    np.random.seed(42)
    x, y = np.random.randn(2, 500)

    fig, ax = plt.subplots()
    fill_q5, line_q5 = kde_contour_plot_2d(ax, x, y, q=5)
    fill_q1, line_q1 = kde_contour_plot_2d(ax, x, y, q=1)
    # The interior iso-probability levels should match bit-identically;
    # only the top value (``vmax``) depends on the plotting window.
    assert_array_equal(fill_q5.levels[:-1], fill_q1.levels[:-1])
    assert_array_equal(line_q5.levels[:-1], line_q1.levels[:-1])


def test_kde_plot_nplot():
    fig, ax = plt.subplots()
    np.random.seed(0)
    data = np.random.randn(1000)
    line, = kde_plot_1d(ax, data, ncompress=1000, nplot_1d=200)
    assert line.get_xdata().size == 200

    fig, ax = plt.subplots()
    np.random.seed(0)
    data_x = np.random.randn(1000)
    data_y = np.random.randn(1000)
    kde_contour_plot_2d(ax, data_x, data_y, ncompress=1000, nplot_2d=900)


@pytest.mark.parametrize('contour_plot_2d',
                         [kde_contour_plot_2d,
                          skipif_no_fastkde(fastkde_contour_plot_2d)])
@pytest.mark.parametrize('levels', [[0.9],
                                    [0.9, 0.6],
                                    [0.9, 0.6, 0.3],
                                    [0.9, 0.7, 0.5, 0.3]])
def test_contour_plot_2d_levels(contour_plot_2d, levels):
    np.random.seed(42)
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    cmap = plt.cm.viridis

    fig, (ax1, ax2) = plt.subplots(2)
    contour_plot_2d(ax1, x, y, levels=levels, cmap=cmap)
    contour_plot_2d(ax2, x, y, levels=levels, cmap=cmap, fc=None)

    # assert that color between filled and unfilled contours matches
    if version.parse(matplotlib.__version__) >= version.parse('3.8.0'):
        color1 = ax1.collections[0].get_facecolor()  # filled face color
        color2 = ax2.collections[0].get_edgecolor()  # unfilled line color
        # first level
        assert_array_equal(color1[0], color2[0])
        # last level
        assert_array_equal(color1[len(levels)-1], color2[len(levels)-1])
    else:
        # first level
        color1 = ax1.collections[0].get_facecolor()  # filled face color
        color2 = ax2.collections[0].get_edgecolor()  # unfilled line color
        assert_array_equal(color1, color2)
        # last level
        color1 = ax1.collections[len(levels)-1].get_facecolor()
        color2 = ax2.collections[len(levels)-1].get_edgecolor()
        assert_array_equal(color1, color2)


def _contour_vertices(contf):
    """Extract all contour vertices from a ContourSet."""
    polygons = [np.concatenate(p) for p in contf.allsegs if p]
    return np.concatenate(polygons)


def _pow10(*x, scale):
    """Exponentiate log-axis test data to displayed coordinates."""
    return tuple(10.0**np.asarray(_) if scale == 'log' else _ for _ in x)


def _log10(x, scale):
    """Log-transform displayed coordinates back to test coordinates."""
    return np.log10(x) if scale == 'log' else x


@pytest.mark.parametrize('scale', ['linear', 'log'])
def test_kde_contour_plot_2d_degenerate(scale):
    np.random.seed(42)
    x = np.random.normal(size=1000)

    # Collinear input (y = 5x): contours narrow perpendicular to the diagonal.
    f = 5
    norm = np.hypot(1, f)
    fig, ax = plt.subplots()
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    contf, _ = kde_contour_plot_2d(ax, *_pow10(x, f*x.copy(), scale=scale))
    assert isinstance(contf, ContourSet)
    vertices = _log10(_contour_vertices(contf), scale)
    perp = (f * vertices[:, 0] - vertices[:, 1]) / norm
    para = (vertices[:, 0] + f * vertices[:, 1]) / norm
    assert np.ptp(perp) < 0.1 * norm / np.sqrt(2)
    assert np.ptp(para) > 4.0 * norm / np.sqrt(2)

    # Collinear input (y = -x): contours narrow perpendicular to the diagonal.
    fig, ax = plt.subplots()
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    contf, _ = kde_contour_plot_2d(ax, *_pow10(x, -x.copy(), scale=scale))
    assert isinstance(contf, ContourSet)
    vertices = _log10(_contour_vertices(contf), scale)
    perp = (vertices[:, 0] + vertices[:, 1]) / np.sqrt(2)
    para = (vertices[:, 0] - vertices[:, 1]) / np.sqrt(2)
    assert np.ptp(perp) < 0.1
    assert np.ptp(para) > 4.0

    # Constant y, viewLim set via set_ylim: thin horizontal band at y=0.5.
    fig, ax = plt.subplots()
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlim(*_pow10((-3, 3), scale=scale))
    ax.set_ylim(*_pow10((-1, 2), scale=scale))
    contf, _ = kde_contour_plot_2d(
        ax, *_pow10(x, np.full_like(x, 0.5), scale=scale)
    )
    assert isinstance(contf, ContourSet)
    vertices = _log10(_contour_vertices(contf), scale)
    assert_allclose(vertices[:, 1], 0.5, atol=0.01)
    assert np.mean(vertices[:, 1]) == pytest.approx(0.5, abs=0.001)
    assert np.ptp(vertices[:, 1]) < 0.1
    assert np.ptp(vertices[:, 0]) > 4.0

    # Constant x, dataLim set by prior plot: thin vertical band at x=0.5.
    fig, ax = plt.subplots()
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.plot(*_pow10([-1, 2], [-3, 3], scale=scale), alpha=0)
    contf, _ = kde_contour_plot_2d(
        ax, *_pow10(np.full_like(x, 0.5), x, scale=scale)
    )
    assert isinstance(contf, ContourSet)
    vertices = _log10(_contour_vertices(contf), scale)
    assert_allclose(vertices[:, 0], 0.5, atol=0.01)
    assert np.mean(vertices[:, 0]) == pytest.approx(0.5, abs=0.001)
    assert np.ptp(vertices[:, 0]) < 0.1
    assert np.ptp(vertices[:, 1]) > 4.0

    # Constant y, no limits: should raise ValueError.
    fig, ax = plt.subplots()
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    with pytest.raises(ValueError, match="`y` variable has zero variance"):
        kde_contour_plot_2d(ax, *_pow10(x, np.full_like(x, 0.5), scale=scale))
    fig, axes = make_2d_axes(['a', 'b'], diagonal=False, upper=False)
    axes['a']['b'].set_xscale(scale)
    axes['a']['b'].set_yscale(scale)
    with pytest.raises(ValueError, match="`b` variable has zero variance"):
        kde_contour_plot_2d(
            axes['a']['b'], *_pow10(x, np.full_like(x, 0.5), scale=scale)
        )

    # Constant x, no limits: should raise ValueError.
    fig, ax = plt.subplots()
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    with pytest.raises(ValueError, match="`x` variable has zero variance"):
        kde_contour_plot_2d(ax, *_pow10(np.full_like(x, 0.5), x, scale=scale))
    fig, axes = make_2d_axes(['a', 'b'], diagonal=False, upper=False)
    axes['a']['b'].set_xscale(scale)
    axes['a']['b'].set_yscale(scale)
    with pytest.raises(ValueError, match="`a` variable has zero variance"):
        kde_contour_plot_2d(
            axes['a']['b'], *_pow10(np.full_like(x, 0.5), x, scale=scale)
        )

    # Highly correlated input (y ~ x) with window-spanning low-weight samples.
    np.random.seed(42)
    sigma = 0.4
    rho = 0.9999
    cov = sigma**2 * np.array([[1.0, rho], [rho, 1.0]])
    s = correlated_gaussian(100, mean=[0.1, 0.0], cov=cov, columns=['a', 'b'])
    fig, ax = plt.subplots()
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    contf, _ = kde_contour_plot_2d(
        ax, *_pow10(s.a, s.b, scale=scale), weights=s.get_weights()
    )
    assert isinstance(contf, ContourSet)
    vertices = _log10(_contour_vertices(contf), scale)
    assert vertices.size > 0
    perp = (vertices[:, 0] - vertices[:, 1]) / np.sqrt(2)
    para = (vertices[:, 0] + vertices[:, 1]) / np.sqrt(2)
    assert np.ptp(perp) < 0.1
    assert np.ptp(para) > 0.7


@pytest.mark.parametrize('axis_aligned,rotated,parallel',
                         [(None, 45, (0, 180)),
                          (0, (45, 0), (0, 1e-10))])
def test_kde_contour_plot_2d_grid_angle(axis_aligned, rotated, parallel):
    # Masking to x - y > 0 creates a hard density edge along y = x.
    np.random.seed(42)
    x = np.random.normal(size=2000)
    y = np.random.normal(size=2000)
    mask = (x - y) > 0
    x, y = x[mask], y[mask]

    # The default axis-aligned grid leaks across the edge by about a bandwidth.
    fig, ax = plt.subplots()
    contf_def, _ = kde_contour_plot_2d(ax, x, y, grid_angle=axis_aligned)
    v_def = _contour_vertices(contf_def)
    assert (v_def[:, 0] - v_def[:, 1]).min() < -0.1

    # Aligned grid_angle confines contour vertices to the x - y > 0 half-plane.
    fig, ax = plt.subplots()
    contf_ang, _ = kde_contour_plot_2d(ax, x, y, grid_angle=rotated)
    v_ang = _contour_vertices(contf_ang)
    assert (v_ang[:, 0] - v_ang[:, 1]).min() == approx((x-y).min(), abs=1e-14)
    assert (v_ang[:, 0] - v_ang[:, 1]).min() >= 0

    # (Near-)parallel angles raise an error.
    with pytest.raises(ValueError,
                       match=fr"grid_angle major \({parallel[0]}\) and minor "
                             fr"\({parallel[1]}\) axes are \(near-\)parallel"):
        kde_contour_plot_2d(ax, x, y, grid_angle=parallel)


@pytest.mark.parametrize('grid_angle', [0, 90, 180, (0, 90), (90, 180)])
def test_kde_contour_plot_2d_grid_angle_axis_aligned(grid_angle):
    # Axis-aligned angles zero out one basis-vector component.
    # Must run without divide-by-zero warnings.
    np.random.seed(42)
    x = np.random.normal(size=500)
    y = np.random.normal(size=500) + 0.3 * x
    fig, ax = plt.subplots()
    with catch_warnings():
        filterwarnings('error', category=RuntimeWarning)
        contf, _ = kde_contour_plot_2d(ax, x, y, grid_angle=grid_angle)
    assert isinstance(contf, ContourSet)


@pytest.mark.parametrize('grid_angle', [0, 90])
def test_basis_aligned_grid_axis_aligned_rows_span_edges(grid_angle):
    # Axis-aligned angles share rectangle corner u-values; the first
    # and last grid rows must span the full edge, not collapse.
    np.random.seed(42)
    data_x = np.random.normal(size=200)
    data_y = np.random.normal(size=200)
    X, Y = _basis_aligned_grid(data_x, data_y, eig=None, ngrid=10,
                               xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0,
                               grid_angle=grid_angle)
    span, const = (X, Y) if grid_angle == 0 else (Y, X)
    assert span[0].min() == approx(-5.0)
    assert span[0].max() == approx(5.0)
    assert const[0].min() == approx(const[0].max())
    assert span[-1].min() == approx(-5.0)
    assert span[-1].max() == approx(5.0)
    assert const[-1].min() == approx(const[-1].max())


@pytest.mark.parametrize('scale', ['linear', 'log'])
def test_plot_window(scale):
    np.random.seed(42)

    # viewLim set via set_xlim and set_ylim.
    fig, ax = plt.subplots()
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlim(*_pow10((-5, 5), scale=scale))
    ax.set_ylim(*_pow10((0, 4), scale=scale))
    assert _plot_window(ax, 'x') == pytest.approx(10)
    assert _plot_window(ax, 'y') == pytest.approx(4)

    # dataLim set by prior plot.
    fig, ax = plt.subplots()
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.plot(*_pow10([0, 10], [0, 1], scale=scale))
    assert _plot_window(ax, 'x') == pytest.approx(10)
    assert _plot_window(ax, 'y') == pytest.approx(1)

    # fresh mpl axis, no limits: raises ValueError.
    fig, ax = plt.subplots()
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    with pytest.raises(ValueError, match=r"ax\.set_xlim"):
        _plot_window(ax, 'x')
    with pytest.raises(ValueError, match=r"ax\.set_ylim"):
        _plot_window(ax, 'y')

    # fresh anesthetic axis, no limits: raises ValueError.
    fig, axes = make_2d_axes(['a', 'b'], diagonal=False, upper=False)
    axes['a']['b'].set_xscale(scale)
    axes['a']['b'].set_yscale(scale)
    with pytest.raises(ValueError, match=r"axes\['a'\]\['b'\]\.set_xlim"):
        _plot_window(axes['a']['b'], 'x')
    with pytest.raises(ValueError, match=r"axes\['a'\]\['b'\]\.set_ylim"):
        _plot_window(axes['a']['b'], 'y')


def test_scatter_plot_2d():
    fig, ax = plt.subplots()
    np.random.seed(2)
    data_x = np.random.randn(100)
    data_y = np.random.randn(100)
    lines, = scatter_plot_2d(ax, data_x, data_y)
    assert isinstance(lines, Line2D)

    fig, ax = plt.subplots()
    points, = scatter_plot_2d(ax, data_x, data_y, color='C0', lw=1)
    assert (points.get_color() == 'C0')
    points, = scatter_plot_2d(ax, data_x, data_y, cmap=plt.cm.viridis)
    assert (points.get_color() == plt.cm.viridis(0.68))
    points, = scatter_plot_2d(ax, data_x, data_y, c='C0', fc='C1', ec='C2')
    assert (points.get_color() == 'C0')
    assert (points.get_markerfacecolor() == 'C1')
    assert (points.get_markeredgecolor() == 'C2')

    # Check that q is ignored
    fig, ax = plt.subplots()
    scatter_plot_2d(ax, data_x, data_y, q=0)


def test_make_axes_logscale():
    # 1d
    fig, axes = make_1d_axes(['x0', 'x1', 'x2', 'x3'], logx=['x1', 'x3'])
    assert axes.loc['x0'].get_xscale() == 'linear'
    assert axes.loc['x1'].get_xscale() == 'log'
    assert axes.loc['x2'].get_xscale() == 'linear'
    assert axes.loc['x3'].get_xscale() == 'log'

    # 2d, logx only
    fig, axes = make_2d_axes(['x0', 'x1', 'x2', 'x3'],
                             logx=['x1', 'x3'])
    for y, rows in axes.iterrows():
        for x, ax in rows.items():
            if x in ['x0', 'x2']:
                assert ax.get_xscale() == 'linear'
            else:
                assert ax.get_xscale() == 'log'
            assert ax.get_yscale() == 'linear'
            with catch_warnings():
                filterwarnings('error', category=UserWarning,
                               message="Attempt to set non-positive")
                ax.set_ylim(-1, 1)

    # 2d, logy only
    fig, axes = make_2d_axes(['x0', 'x1', 'x2', 'x3'],
                             logy=['x1', 'x3'])
    for y, rows in axes.iterrows():
        for x, ax in rows.items():
            assert ax.get_xscale() == 'linear'
            with catch_warnings():
                filterwarnings('error', category=UserWarning,
                               message="Attempt to set non-positive")
                ax.set_xlim(-1, 1)
            if y in ['x0', 'x2']:
                assert ax.get_yscale() == 'linear'
            else:
                assert ax.get_yscale() == 'log'

    # 2d, logx and logy
    fig, axes = make_2d_axes(['x0', 'x1', 'x2', 'x3'],
                             logx=['x1', 'x3'],
                             logy=['x1', 'x3'])
    for y, rows in axes.iterrows():
        for x, ax in rows.items():
            if x in ['x0', 'x2']:
                assert ax.get_xscale() == 'linear'
            else:
                assert ax.get_xscale() == 'log'
            if y in ['x0', 'x2']:
                assert ax.get_yscale() == 'linear'
            else:
                assert ax.get_yscale() == 'log'


@pytest.mark.parametrize('plot_1d', [kde_plot_1d,
                                     skipif_no_fastkde(fastkde_plot_1d),
                                     hist_plot_1d])
def test_logscale_1d(plot_1d):
    np.random.seed(42)
    logdata = np.random.randn(1000)
    data = 10**logdata

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    p = plot_1d(ax, data)
    if 'kde' in plot_1d.__name__:
        amax = abs(np.log10(p[0].get_xdata()[np.argmax(p[0].get_ydata())]))
    else:
        amax = abs(np.log10(p[1][np.argmax(p[0])]))
    assert amax < 0.5


@pytest.mark.parametrize('b', ['scott', 20, np.logspace(-5, 5, 20)])
def test_logscale_hist_kwargs(b):
    np.random.seed(42)
    logdata = np.random.randn(1000)
    data = 10**logdata

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    h, edges, _ = hist_plot_1d(ax, data, bins=b)
    amax = abs(np.log10(edges[np.argmax(h)]))
    assert amax < 0.5
    assert edges[0] < 1e-3
    assert edges[-1] > 1e3
    h, edges, _ = hist_plot_1d(ax, data, bins=b, range=None)
    amax = abs(np.log10(edges[np.argmax(h)]))
    assert amax < 0.5
    assert edges[0] < 1e-3
    assert edges[-1] > 1e3
    h, edges, _ = hist_plot_1d(ax, data, bins=b, range=(1e-3, 1e3))
    amax = abs(np.log10(edges[np.argmax(h)]))
    assert amax < 0.5
    if isinstance(b, (int, str)):
        # edges are trimmed according to range
        assert edges[0] == 1e-3
        assert edges[-1] == 1e3
    else:
        # edges passed directly to bins are not trimmed according to range
        assert edges[0] == b[0]
        assert edges[-1] == b[-1]


@pytest.mark.parametrize('plot_2d',
                         [kde_contour_plot_2d,
                          skipif_no_fastkde(fastkde_contour_plot_2d),
                          hist_plot_2d, scatter_plot_2d])
def test_logscale_2d(plot_2d):
    np.random.seed(0)
    logx = np.random.randn(1000)
    logy = np.random.randn(1000)
    x = 10**logx
    y = 10**logy

    # logx
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    p = plot_2d(ax, x, logy)
    if 'kde' in plot_2d.__name__:
        if version.parse(matplotlib.__version__) >= version.parse('3.8.0'):
            xmax, ymax = p[0].get_paths()[1].vertices.T
        else:
            xmax, ymax = p[0].allsegs[1][0].T
        xmax = np.mean(np.log10(xmax))
        ymax = np.mean(ymax)
    elif 'hist' in plot_2d.__name__:
        c = p.get_coordinates()
        c = (c[:-1, :] + c[1:, :]) / 2
        c = (c[:, :-1] + c[:, 1:]) / 2
        c = c.reshape((-1, 2))
        xmax = abs(np.log10(c[np.argmax(p.get_array())][0]))
        ymax = abs(c[np.argmax(p.get_array())][1])
    else:
        xmax = np.mean(np.log10(p[0].get_xdata()))
        ymax = np.mean(p[0].get_ydata())
    assert xmax < 0.5
    assert ymax < 0.5

    # logy
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    p = plot_2d(ax, logx, y)
    if 'kde' in plot_2d.__name__:
        if version.parse(matplotlib.__version__) >= version.parse('3.8.0'):
            xmax, ymax = p[0].get_paths()[1].vertices[0].T
        else:
            xmax, ymax = p[0].allsegs[1][0].T
        xmax = np.mean(xmax)
        ymax = np.mean(np.log10(ymax))
    elif 'hist' in plot_2d.__name__:
        c = p.get_coordinates()
        c = (c[:-1, :] + c[1:, :]) / 2
        c = (c[:, :-1] + c[:, 1:]) / 2
        c = c.reshape((-1, 2))
        xmax = abs(c[np.argmax(p.get_array())][0])
        ymax = abs(np.log10(c[np.argmax(p.get_array())][1]))
    else:
        xmax = np.mean(p[0].get_xdata())
        ymax = np.mean(np.log10(p[0].get_ydata()))
    assert xmax < 0.5
    assert ymax < 0.5

    # logx and logy
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    p = plot_2d(ax, x, y)
    if 'kde' in plot_2d.__name__:
        if version.parse(matplotlib.__version__) >= version.parse('3.8.0'):
            xmax, ymax = p[0].get_paths()[1].vertices[0].T
        else:
            xmax, ymax = p[0].allsegs[1][0].T
        xmax = np.mean(np.log10(xmax))
        ymax = np.mean(np.log10(ymax))
    elif 'hist' in plot_2d.__name__:
        c = p.get_coordinates()
        c = (c[:-1, :] + c[1:, :]) / 2
        c = (c[:, :-1] + c[:, 1:]) / 2
        c = c.reshape((-1, 2))
        xmax = abs(np.log10(c[np.argmax(p.get_array())][0]))
        ymax = abs(np.log10(c[np.argmax(p.get_array())][1]))
    else:
        xmax = np.mean(np.log10(p[0].get_xdata()))
        ymax = np.mean(np.log10(p[0].get_ydata()))
    assert xmax < 0.5
    assert ymax < 0.5


@pytest.mark.parametrize('sigmas', [(1, '1sigma', 0.682689492137086),
                                    (2, '2sigma', 0.954499736103642),
                                    (3, '3sigma', 0.997300203936740),
                                    (4, '4sigma', 0.999936657516334),
                                    (5, '5sigma', 0.999999426696856)])
def test_quantile_plot_interval_str(sigmas):
    qi1, qi2 = quantile_plot_interval(q=sigmas[0])
    qs1, qs2 = quantile_plot_interval(q=sigmas[1])
    assert qi1 == pytest.approx(0.5 - sigmas[2] / 2)
    assert qi2 == pytest.approx(0.5 + sigmas[2] / 2)
    assert qs1 == pytest.approx(0.5 - sigmas[2] / 2)
    assert qs2 == pytest.approx(0.5 + sigmas[2] / 2)


@pytest.mark.parametrize('floats', [0, 0.1, 0.9])
def test_quantile_plot_interval_float(floats):
    q1, q2 = quantile_plot_interval(q=floats)
    assert q1 == min(floats, 1 - floats)
    assert q2 == max(floats, 1 - floats)


@pytest.mark.parametrize('q1, q2', [(0, 1), (0.1, 0.9), (0, 0.9), (0.1, 1)])
def test_quantile_plot_interval_tuple(q1, q2):
    _q1, _q2 = quantile_plot_interval(q=(q1, q2))
    assert _q1 == q1
    assert _q2 == q2


@pytest.mark.parametrize('color', ['C0', 'k', 'gold', '#00FFFF',
                                   (1.0, 1.0, 0.0, 1.0)])
def test_basic_cmap(color):
    cmap = basic_cmap(color)

    # Check that the basic cmap is reversible
    cmap.reversed()
