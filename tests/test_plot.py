import matplotlib_agg  # noqa: F401
import pytest
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from anesthetic.plot import (make_1d_axes, make_2d_axes, kde_plot_1d,
                             fastkde_plot_1d, hist_plot_1d, hist_plot_2d,
                             fastkde_contour_plot_2d, kde_contour_plot_2d,
                             scatter_plot_2d, quantile_plot_interval,
                             basic_cmap)
from numpy.testing import assert_array_equal

from matplotlib.contour import QuadContourSet
from matplotlib.tri import TriContourSet
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.colors import ColorConverter, to_rgba
from matplotlib.figure import Figure
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from scipy.special import erf
from scipy.interpolate import interp1d


def test_make_1d_axes():
    paramnames = ['A', 'B', 'C', 'D', 'E']
    tex = {'A': 'tA', 'B': 'tB', 'C': 'tC', 'D': 'tD', 'E': 'tE'}

    # Check no optional arguments
    fig, axes = make_1d_axes(paramnames)
    assert(isinstance(fig, Figure))
    assert(isinstance(axes, Series))
    assert_array_equal(axes.index, paramnames)
    for p, ax in axes.iteritems():
        assert(ax.get_xlabel() == p)

    # Check tex argument
    fig, axes = make_1d_axes(paramnames, tex=tex)
    for t in tex:
        assert(axes[t].get_xlabel() != t)
        assert(axes[t].get_xlabel() == tex[t])

    # Check fig argument
    fig0 = plt.figure()
    fig0.suptitle('hi there')
    fig, axes = make_1d_axes(paramnames)
    assert(fig is not fig0)
    fig, axes = make_1d_axes(paramnames, fig=fig0)
    assert(fig is fig0)

    # Check ncols argument
    fig, axes = make_1d_axes(paramnames, ncols=2)
    nrows, ncols = axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert(ncols == 2)

    # Check gridspec argument
    grid = gs.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
    g00 = grid[0, 0]
    fig, axes = make_1d_axes(paramnames, subplot_spec=g00)
    assert(g00 is axes[0].get_subplotspec().get_topmost_subplotspec())

    # Check unexpected kwargs
    with pytest.raises(TypeError):
        make_1d_axes(paramnames, foo='bar')
    plt.close("all")


def test_make_2d_axes_inputs_outputs():
    paramnames_x = ['A', 'B', 'C', 'D']
    paramnames_y = ['B', 'A', 'D', 'E']

    # 2D axes
    fig, axes = make_2d_axes([paramnames_x, paramnames_y])
    assert(isinstance(fig, Figure))
    assert(isinstance(axes, DataFrame))
    assert_array_equal(axes.index, paramnames_y)
    assert_array_equal(axes.columns, paramnames_x)

    # Axes labels
    for p, ax in axes.iloc[:, 0].iteritems():
        assert(ax.get_ylabel() == p)

    for p, ax in axes.iloc[-1].iteritems():
        assert(ax.get_xlabel() == p)

    for ax in axes.iloc[:-1, 1:].to_numpy().flatten():
        assert(ax.get_xlabel() == '')
        assert(ax.get_ylabel() == '')

    # Check fig argument
    fig0 = plt.figure()
    fig, axes = make_2d_axes(paramnames_x)
    assert(fig is not fig0)
    fig, axes = make_2d_axes(paramnames_x, fig=fig0)
    assert(fig is fig0)
    plt.close("all")

    # Check gridspec argument
    grid = gs.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
    g00 = grid[0, 0]
    fig, axes = make_2d_axes(paramnames_x, subplot_spec=g00)
    assert(g00 is axes.iloc[0, 0].get_subplotspec().get_topmost_subplotspec())

    # Check unexpected kwargs
    with pytest.raises(TypeError):
        make_2d_axes(paramnames_x, foo='bar')


def test_make_2d_axes_behaviour():
    np.random.seed(0)

    def calc_n(axes):
        """Compute the number of upper, lower and diagonal plots."""
        n = {'upper': 0, 'lower': 0, 'diagonal': 0}
        for y, row in axes.iterrows():
            for x, ax in row.iteritems():
                if ax is not None:
                    n[ax.position] += 1
        return n

    # Check for only paramnames_x
    paramnames_x = ['A', 'B', 'C', 'D']
    nx = len(paramnames_x)
    for upper in [True, False]:
        for lower in [True, False]:
            for diagonal in [True, False]:
                fig, axes = make_2d_axes(paramnames_x,
                                         upper=upper,
                                         lower=lower,
                                         diagonal=diagonal)
                ns = calc_n(axes)
                assert(ns['upper'] == upper * nx*(nx-1)//2)
                assert(ns['lower'] == lower * nx*(nx-1)//2)
                assert(ns['diagonal'] == diagonal * nx)

    plt.close("all")

    for paramnames_y in [['A', 'B', 'C', 'D'],
                         ['A', 'C', 'B', 'D'],
                         ['D', 'C', 'B', 'A'],
                         ['C', 'B', 'A'],
                         ['E', 'F', 'G', 'H'],
                         ['A', 'B', 'E', 'F'],
                         ['B', 'E', 'A', 'F'],
                         ['B', 'F', 'A', 'H', 'G'],
                         ['B', 'A', 'H', 'G']]:
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

        for upper in [True, False]:
            for lower in [True, False]:
                for diagonal in [True, False]:
                    fig, axes = make_2d_axes(params,
                                             upper=upper,
                                             lower=lower,
                                             diagonal=diagonal)
                    ns = calc_n(axes)
                    assert(ns['upper'] == upper * nu)
                    assert(ns['lower'] == lower * nl)
                    assert(ns['diagonal'] == diagonal * nd)
        plt.close("all")


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
            for j, ax in row.iteritems():
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
        plt.close("all")
    with pytest.raises(ValueError):
        make_2d_axes(paramnames, upper=upper, ticks='spam')


def test_2d_axes_limits():
    np.random.seed(0)
    paramnames = ['A', 'B', 'C', 'D']
    fig, axes = make_2d_axes(paramnames)
    for x in paramnames:
        for y in paramnames:
            a, b, c, d = np.random.rand(4)
            axes[x][y].set_xlim(a, b)
            for z in paramnames:
                assert(axes[x][z].get_xlim() == (a, b))
                assert(axes[z][x].get_ylim() == (a, b))

            axes[x][y].set_ylim(c, d)
            for z in paramnames:
                assert(axes[y][z].get_xlim() == (c, d))
                assert(axes[z][y].get_ylim() == (c, d))


@pytest.mark.parametrize('axesparams', [['A', 'B', 'C', 'D'],
                                        [['A', 'B', 'C', 'D'], ['A', 'B']],
                                        [['A', 'B'], ['A', 'B', 'C', 'D']]])
@pytest.mark.parametrize('params, values', [('A', 0),
                                            (['A', 'C', 'E'], [0, 0, 0]),
                                            (['A', 'C', 'C'], [0, 0, 0.5])])
@pytest.mark.parametrize('upper', [True, False])
def test_2d_axlines_axspans(axesparams, params, values, upper):
    values = np.array(values)
    line_kwargs = dict(c='k', ls='--', lw=0.5)
    span_kwargs = dict(c='k', alpha=0.5)
    fig, axes = make_2d_axes(axesparams, upper=upper)
    axes.axlines(params, values, **line_kwargs)
    axes.axspans(params, values-0.1, values+0.1, **span_kwargs)
    plt.close("all")


@pytest.mark.parametrize('params, values', [('A', [0, 0]),
                                            (['A', 'C'], 0),
                                            (['A', 'C'], [0, 0, 0.5]),
                                            (['A', 'C', 'C'], [0, 0])])
def test_2d_axlines_axspans_error(params, values):
    values = np.array(values)
    axesparams = ['A', 'B', 'C', 'D']
    fig, axes = make_2d_axes(axesparams)
    with pytest.raises(ValueError):
        axes.axlines(params, values)
    with pytest.raises(ValueError):
        axes.axspans(params, values-0.1, values+0.1)
    plt.close("all")


@pytest.mark.parametrize('plot_1d', [kde_plot_1d, fastkde_plot_1d])
def test_kde_plot_1d(plot_1d):
    fig, ax = plt.subplots()
    np.random.seed(0)
    data = np.random.randn(1000)

    try:
        # Check height
        line, = plot_1d(ax, data)
        assert(isinstance(line, Line2D))
        assert(line.get_ydata().max() <= 1)

        # Check arguments are passed onward to underlying function
        line, = plot_1d(ax, data, color='r')
        assert(line.get_color() == 'r')
        line, = plot_1d(ax, data, cmap=plt.cm.Blues)
        assert(line.get_color() == plt.cm.Blues(0.68))

        # Check xmin
        xmin = -0.5
        line, = plot_1d(ax, data, xmin=xmin)
        assert((line.get_xdata() >= xmin).all())

        # Check xmax
        xmax = 0.5
        line, = plot_1d(ax, data, xmax=xmax)
        assert((line.get_xdata() <= xmax).all())

        # Check xmin and xmax
        line, = plot_1d(ax, data, xmin=xmin, xmax=xmax)
        assert((line.get_xdata() <= xmax).all())
        assert((line.get_xdata() >= xmin).all())
        plt.close("all")

        # Check q
        plot_1d(ax, data, q='1sigma')
        plot_1d(ax, data, q=0)
        plot_1d(ax, data, q=1)
        plot_1d(ax, data, q=0.1)
        plot_1d(ax, data, q=0.9)
        plot_1d(ax, data, q=(0.1, 0.9))

        # Check iso-probability code
        line, fill = plot_1d(ax, data, facecolor=True)
        plot_1d(ax, data, facecolor=True, levels=[0.8, 0.6, 0.2])
        line, fill = plot_1d(ax, data, fc='blue', color='k', ec='r')
        assert(np.all(fill[0].get_edgecolor()[0] == to_rgba('r')))
        assert (to_rgba(line[0].get_color()) == to_rgba('r'))
        line, fill = plot_1d(ax, data, fc=True, color='k', ec=None)
        assert(len(fill[0].get_edgecolor()) == 0)
        assert (to_rgba(line[0].get_color()) == to_rgba('k'))
        plt.close("all")

        # Check levels
        with pytest.raises(ValueError):
            ax = plt.gca()
            plot_1d(ax, data, fc=True, levels=[0.68, 0.95])
            plt.close("all")

        # Check xlim, Gaussian (i.e. limits reduced to relevant data range)
        fig, ax = plt.subplots()
        data = np.random.randn(1000) * 0.01 + 0.5
        plot_1d(ax, data, xmin=0, xmax=1)
        xmin, xmax = ax.get_xlim()
        assert(xmin > 0.4)
        assert(xmax < 0.6)
        plt.close("all")
        # Check xlim, Uniform (i.e. data and limits span entire prior boundary)
        fig, ax = plt.subplots()
        data = np.random.uniform(size=1000)
        plot_1d(ax, data, xmin=0, xmax=1)
        xmin, xmax = ax.get_xlim()
        assert(xmin == 0)
        assert(xmax == 1)
        plt.close("all")

    except ImportError:
        if 'fastkde' not in sys.modules:
            pass


def test_hist_plot_1d():
    fig, ax = plt.subplots()
    np.random.seed(0)
    data = np.random.randn(1000)
    for p in ['', 'astropyhist']:
        try:
            # Check heights for histtype 'bar'
            bars = hist_plot_1d(ax, data, histtype='bar', plotter=p)
            assert(np.all([isinstance(b, Patch) for b in bars]))
            assert(max([b.get_height() for b in bars]) == 1.)
            assert(np.all(np.array([b.get_height() for b in bars]) <= 1.))

            # Check heights for histtype 'step'
            polygon, = hist_plot_1d(ax, data, histtype='step', plotter=p)
            assert(isinstance(polygon, Polygon))
            trans = polygon.get_transform() - ax.transData
            assert(np.isclose(trans.transform(polygon.xy)[:, -1].max(), 1.,
                              rtol=1e-10, atol=1e-10))
            assert(np.all(trans.transform(polygon.xy)[:, -1] <= 1.))

            # Check heights for histtype 'stepfilled'
            polygon, = hist_plot_1d(ax, data, histtype='stepfilled', plotter=p)
            assert(isinstance(polygon, Polygon))
            trans = polygon.get_transform() - ax.transData
            assert(np.isclose(trans.transform(polygon.xy)[:, -1].max(), 1.,
                              rtol=1e-10, atol=1e-10))
            assert(np.all(trans.transform(polygon.xy)[:, -1] <= 1.))

            # Check arguments are passed onward to underlying function
            bars = hist_plot_1d(ax, data, histtype='bar',
                                color='r', alpha=0.5, plotter=p)
            cc = ColorConverter.to_rgba('r', alpha=0.5)
            assert(np.all([b.get_fc() == cc for b in bars]))
            bars = hist_plot_1d(ax, data, histtype='bar',
                                cmap=plt.cm.viridis, alpha=0.5, plotter=p)
            cc = ColorConverter.to_rgba(plt.cm.viridis(0.68), alpha=0.5)
            assert(np.all([b.get_fc() == cc for b in bars]))
            polygon, = hist_plot_1d(ax, data, histtype='step',
                                    color='r', alpha=0.5, plotter=p)
            assert(polygon.get_ec() == ColorConverter.to_rgba('r', alpha=0.5))
            polygon, = hist_plot_1d(ax, data, histtype='step',
                                    cmap=plt.cm.viridis, color='r', plotter=p)
            assert(polygon.get_ec() == ColorConverter.to_rgba('r'))

            # Check xmin
            for xmin in [-np.inf, -0.5]:
                bars = hist_plot_1d(ax, data, histtype='bar',
                                    xmin=xmin, plotter=p)
                assert((np.array([b.xy[0] for b in bars]) >= xmin).all())
                polygon, = hist_plot_1d(ax, data, histtype='step', xmin=xmin)
                assert((polygon.xy[:, 0] >= xmin).all())

            # Check xmax
            for xmax in [np.inf, 0.5]:
                bars = hist_plot_1d(ax, data, histtype='bar',
                                    xmax=xmax, plotter=p)
                assert((np.array([b.xy[-1] for b in bars]) <= xmax).all())
                polygon, = hist_plot_1d(ax, data, histtype='step',
                                        xmax=xmax, plotter=p)
                assert((polygon.xy[:, 0] <= xmax).all())

            # Check xmin and xmax
            bars = hist_plot_1d(ax, data, histtype='bar',
                                xmin=xmin, xmax=xmax, plotter=p)
            assert((np.array([b.xy[0] for b in bars]) >= -0.5).all())
            assert((np.array([b.xy[-1] for b in bars]) <= 0.5).all())
            polygon, = hist_plot_1d(ax, data, histtype='step',
                                    xmin=xmin, xmax=xmax, plotter=p)
            assert((polygon.xy[:, 0] >= -0.5).all())
            assert((polygon.xy[:, 0] <= 0.5).all())
            plt.close("all")
        except ImportError:
            if p == 'astropyhist' and 'astropy' not in sys.modules:
                pass


def test_hist_plot_2d():
    fig, ax = plt.subplots()
    np.random.seed(0)
    data_x, data_y = np.random.randn(2, 10000)
    hist_plot_2d(ax, data_x, data_y)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    assert xmin > -3 and xmax < 3 and ymin > -3 and ymax < 3

    hist_plot_2d(ax, data_x, data_y, xmin=-np.inf)
    hist_plot_2d(ax, data_x, data_y, xmax=np.inf)
    hist_plot_2d(ax, data_x, data_y, ymin=-np.inf)
    hist_plot_2d(ax, data_x, data_y, ymax=np.inf)
    assert xmin > -3 and xmax < 3 and ymin > -3 and ymax < 3

    data_x, data_y = np.random.uniform(-10, 10, (2, 1000000))
    weights = np.exp(-(data_x**2 + data_y**2)/2)
    hist_plot_2d(ax, data_x, data_y, weights=weights, bins=30)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    assert xmin > -3 and xmax < 3 and ymin > -3 and ymax < 3


@pytest.mark.parametrize('plot_1d', [kde_plot_1d, fastkde_plot_1d])
@pytest.mark.parametrize('s', [1, 2])
def test_1d_density_kwarg(plot_1d, s):
    try:
        np.random.seed(0)
        x = np.random.normal(scale=s, size=2000)
        fig, ax = plt.subplots()

        # hist density = False:
        h = hist_plot_1d(ax, x, density=False, bins=np.linspace(-5.5, 5.5, 12))
        bar_height = h.get_children()[len(h.get_children()) // 2].get_height()
        assert(bar_height == pytest.approx(1, rel=0.1))

        # kde density = False:
        k = plot_1d(ax, x, density=False)[0]
        f = interp1d(k.get_xdata(), k.get_ydata(), 'cubic', assume_sorted=True)
        kde_height = f(0)
        assert(kde_height == pytest.approx(1, rel=0.1))

        # hist density = True:
        h = hist_plot_1d(ax, x, density=True, bins=np.linspace(-5.5, 5.5, 12))
        bar_height = h.get_children()[len(h.get_children()) // 2].get_height()
        assert(bar_height == pytest.approx(erf(0.5 / np.sqrt(2) / s), rel=0.1))

        # kde density = True:
        k = plot_1d(ax, x, density=True)[0]
        f = interp1d(k.get_xdata(), k.get_ydata(), 'cubic', assume_sorted=True)
        kde_height = f(0)
        gauss_norm = 1 / np.sqrt(2 * np.pi * s**2)
        assert(kde_height == pytest.approx(gauss_norm, rel=0.1))

        plt.close("all")

    except ImportError:
        if 'fastkde' not in sys.modules:
            pass


@pytest.mark.parametrize('contour_plot_2d', [kde_contour_plot_2d,
                                             fastkde_contour_plot_2d])
def test_contour_plot_2d(contour_plot_2d):
    try:
        ax = plt.gca()
        np.random.seed(1)
        data_x = np.random.randn(1000)
        data_y = np.random.randn(1000)
        cf, ct = contour_plot_2d(ax, data_x, data_y)
        if contour_plot_2d is fastkde_contour_plot_2d:
            assert(isinstance(cf, QuadContourSet))
            assert(isinstance(ct, QuadContourSet))
        elif contour_plot_2d is kde_contour_plot_2d:
            assert(isinstance(cf, TriContourSet))
            assert(isinstance(ct, TriContourSet))

        xmin, xmax, ymin, ymax = -0.5, 0.5, -0.5, 0.5

        # Check xmin
        ax = plt.gca()
        contour_plot_2d(ax, data_x, data_y, xmin=xmin)
        assert(ax.get_xlim()[0] >= xmin)
        plt.close()

        # Check xmax
        ax = plt.gca()
        contour_plot_2d(ax, data_x, data_y, xmax=xmax)
        assert(ax.get_xlim()[1] <= xmax)
        plt.close()

        # Check xmin and xmax
        ax = plt.gca()
        contour_plot_2d(ax, data_x, data_y, xmin=xmin, xmax=xmax)
        assert(ax.get_xlim()[1] <= xmax)
        assert(ax.get_xlim()[0] >= xmin)
        plt.close()

        # Check ymin
        ax = plt.gca()
        contour_plot_2d(ax, data_x, data_y, ymin=ymin)
        assert(ax.get_ylim()[0] >= ymin)
        plt.close()

        # Check ymax
        ax = plt.gca()
        contour_plot_2d(ax, data_x, data_y, ymax=ymax)
        assert(ax.get_ylim()[1] <= ymax)
        plt.close()

        # Check ymin and ymax
        ax = plt.gca()
        contour_plot_2d(ax, data_x, data_y, ymin=ymin, ymax=ymax)
        assert(ax.get_ylim()[1] <= ymax)
        assert(ax.get_ylim()[0] >= ymin)
        plt.close()

        # Check levels
        with pytest.raises(ValueError):
            ax = plt.gca()
            contour_plot_2d(ax, data_x, data_y, levels=[0.68, 0.95])

        # Check q
        ax = plt.gca()
        contour_plot_2d(ax, data_x, data_y, q=0)
        plt.close()

        # Check unfilled
        cmap = basic_cmap('C2')
        ax = plt.gca()
        cf1, ct1 = contour_plot_2d(ax, data_x, data_y, facecolor='C2')
        cf2, ct2 = contour_plot_2d(ax, data_x, data_y, fc='None', cmap=cmap)
        # filled `contourf` and unfilled `contour` colors are the same:
        assert cf1.tcolors[0] == ct2.tcolors[0]
        assert cf1.tcolors[1] == ct2.tcolors[1]
        cf, ct = contour_plot_2d(ax, data_x, data_y, edgecolor='C0')
        assert ct.colors == 'C0'
        cf, ct = contour_plot_2d(ax, data_x, data_y, ec='C0', cmap=plt.cm.Reds)
        assert cf.get_cmap() == plt.cm.Reds
        assert ct.colors == 'C0'
        plt.close("all")
        ax = plt.gca()
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
        plt.close("all")

        # Check limits, Gaussian (i.e. limits reduced to relevant data range)
        fig, ax = plt.subplots()
        data_x = np.random.randn(1000) * 0.01 + 0.5
        data_y = np.random.randn(1000) * 0.01 + 0.5
        contour_plot_2d(ax, data_x, data_y, xmin=0, xmax=1, ymin=0, ymax=1)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        assert(xmin > 0.4)
        assert(xmax < 0.6)
        assert(ymin > 0.4)
        assert(ymax < 0.6)
        plt.close("all")
        # Check limits, Uniform (i.e. data & limits span entire prior boundary)
        fig, ax = plt.subplots()
        data_x = np.random.uniform(size=1000)
        data_y = np.random.uniform(size=1000)
        contour_plot_2d(ax, data_x, data_y, xmin=0, xmax=1, ymin=0, ymax=1)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        assert(xmin == 0)
        assert(xmax == 1)
        assert(ymin == 0)
        assert(ymax == 1)
        plt.close("all")

    except ImportError:
        if 'fastkde' not in sys.modules:
            pass


@pytest.mark.parametrize('contour_plot_2d', [kde_contour_plot_2d,
                                             fastkde_contour_plot_2d])
@pytest.mark.parametrize('levels', [[0.9],
                                    [0.9, 0.6],
                                    [0.9, 0.6, 0.3],
                                    [0.9, 0.7, 0.5, 0.3]])
def test_contour_plot_2d_levels(contour_plot_2d, levels):
    try:
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        cmap = plt.cm.viridis

        ax1 = plt.subplot(211)
        contour_plot_2d(ax1, x, y, levels=levels, cmap=cmap)
        ax2 = plt.subplot(212)
        contour_plot_2d(ax2, x, y, levels=levels, cmap=cmap, fc=None)

        # assert that color between filled and unfilled contours matches
        # first level
        color1 = ax1.collections[0].get_facecolor()  # filled face color
        color2 = ax2.collections[0].get_edgecolor()  # unfilled line color
        assert_array_equal(color1, color2)
        # last level
        color1 = ax1.collections[len(levels)-1].get_facecolor()
        color2 = ax2.collections[len(levels)-1].get_edgecolor()
        assert_array_equal(color1, color2)

        plt.close("all")

    except ImportError:
        if 'fastkde' not in sys.modules:
            pass


def test_scatter_plot_2d():
    fig, ax = plt.subplots()
    np.random.seed(2)
    data_x = np.random.randn(1000)
    data_y = np.random.randn(1000)
    lines, = scatter_plot_2d(ax, data_x, data_y)
    assert(isinstance(lines, Line2D))

    xmin, xmax, ymin, ymax = -0.5, 0.5, -0.5, 0.5
    ax = plt.gca()
    scatter_plot_2d(ax, data_x, data_y, xmin=xmin)
    assert(ax.get_xlim()[0] >= xmin)
    plt.close()

    ax = plt.gca()
    scatter_plot_2d(ax, data_x, data_y, xmax=xmax)
    assert(ax.get_xlim()[1] <= xmax)
    plt.close()

    ax = plt.gca()
    scatter_plot_2d(ax, data_x, data_y, ymin=ymin)
    assert(ax.get_ylim()[0] >= ymin)
    plt.close()

    ax = plt.gca()
    scatter_plot_2d(ax, data_x, data_y, ymax=ymax)
    assert(ax.get_ylim()[1] <= ymax)
    plt.close()

    ax = plt.gca()
    points, = scatter_plot_2d(ax, data_x, data_y, color='C0', lw=1)
    assert (points.get_color() == 'C0')
    points, = scatter_plot_2d(ax, data_x, data_y, cmap=plt.cm.viridis)
    assert (points.get_color() == plt.cm.viridis(0.68))
    points, = scatter_plot_2d(ax, data_x, data_y, c='C0', fc='C1', ec='C2')
    assert (points.get_color() == 'C0')
    assert (points.get_markerfacecolor() == 'C1')
    assert (points.get_markeredgecolor() == 'C2')
    plt.close()

    # Check q
    ax = plt.gca()
    scatter_plot_2d(ax, data_x, data_y, q=0)
    plt.close()


@pytest.mark.parametrize('sigmas', [('1sigma', 0.682689492137086),
                                    ('2sigma', 0.954499736103642),
                                    ('3sigma', 0.997300203936740),
                                    ('4sigma', 0.999936657516334),
                                    ('5sigma', 0.999999426696856)])
def test_quantile_plot_interval_str(sigmas):
    q1, q2 = quantile_plot_interval(q=sigmas[0])
    assert q1 == 0.5 - sigmas[1] / 2
    assert q2 == 0.5 + sigmas[1] / 2


@pytest.mark.parametrize('floats', [0, 1, 0.1, 0.9])
def test_quantile_plot_interval_float(floats):
    q1, q2 = quantile_plot_interval(q=floats)
    assert q1 == min(floats, 1 - floats)
    assert q2 == max(floats, 1 - floats)


@pytest.mark.parametrize('q1, q2', [(0, 1), (0.1, 0.9), (0, 0.9), (0.1, 1)])
def test_quantile_plot_interval_tuple(q1, q2):
    _q1, _q2 = quantile_plot_interval(q=(q1, q2))
    assert _q1 == q1
    assert _q2 == q2
