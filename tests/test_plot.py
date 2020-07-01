import matplotlib_agg  # noqa: F401
import pytest
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from anesthetic.plot import (make_1d_axes, make_2d_axes, kde_plot_1d,
                             fastkde_plot_1d, hist_plot_1d, hist_plot_2d,
                             fastkde_contour_plot_2d, kde_contour_plot_2d,
                             scatter_plot_2d, quantile_plot_interval)
from numpy.testing import assert_array_equal

from matplotlib.contour import QuadContourSet
from matplotlib.tri import TriContourSet
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.colors import ColorConverter
from matplotlib.figure import Figure
from pandas.core.series import Series
from pandas.core.frame import DataFrame


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

    for ax in axes.iloc[:-1, 1:].values.flatten():
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
            polygon, = hist_plot_1d(ax, data, histtype='step',
                                    color='r', alpha=0.5, plotter=p)
            assert(polygon.get_ec() == ColorConverter.to_rgba('r', alpha=0.5))

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


@pytest.mark.parametrize('contour_plot_2d', [kde_contour_plot_2d,
                                             fastkde_contour_plot_2d])
def test_contour_plot_2d(contour_plot_2d):
    try:
        ax = plt.gca()
        np.random.seed(1)
        data_x = np.random.randn(1000)
        data_y = np.random.randn(1000)
        c = contour_plot_2d(ax, data_x, data_y)
        if contour_plot_2d is fastkde_contour_plot_2d:
            assert(isinstance(c, QuadContourSet))
        elif contour_plot_2d is kde_contour_plot_2d:
            assert(isinstance(c, TriContourSet))

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

        # Check q
        ax = plt.gca()
        contour_plot_2d(ax, data_x, data_y, q=0)
        plt.close()
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
