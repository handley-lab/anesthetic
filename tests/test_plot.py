import pytest
import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from anesthetic.plot import make_1D_axes, make_2D_axes, plot_1d, contour_plot_2d, scatter_plot_2d
from numpy.testing import assert_array_equal, assert_array_less

from matplotlib.contour import QuadContourSet
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pandas.core.series import Series
from pandas.core.frame import DataFrame

def test_make_1D_axes():
    paramnames = ['A', 'B', 'C', 'D', 'E']
    tex = {'A':'tA', 'B':'tB', 'C':'tC', 'D':'tD', 'E':'tE'}

    # Check no optional arguments
    fig, axes = make_1D_axes(paramnames)
    assert(isinstance(fig, Figure))
    assert(isinstance(axes, Series))
    assert_array_equal(axes.index, paramnames)
    for p, ax in axes.iteritems():
        assert(ax.get_xlabel() == p)
    plt.close(fig)

    # Check tex argument
    fig, axes = make_1D_axes(paramnames, tex=tex)
    for t in tex:
        assert(axes[t].get_xlabel() != t)
        assert(axes[t].get_xlabel() == tex[t])
    plt.close(fig)

    # Check fig argument
    fig0 = plt.figure()
    fig1 = plt.figure()
    fig, axes = make_1D_axes(paramnames)
    assert(fig is fig1)
    fig, axes = make_1D_axes(paramnames, fig=fig0)
    assert(fig is fig0)
    plt.close(fig0)
    plt.close(fig1)

    # Check ncols argument
    fig, axes = make_1D_axes(paramnames, ncols=2)
    nrows, ncols = axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert(ncols==2)
    plt.close(fig)

    # Check gridspec argument
    grid = gs.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[3,1])
    g00 = grid[0,0]
    fig, axes = make_1D_axes(paramnames, subplot_spec=g00)
    assert(g00 is axes[0].get_subplotspec().get_topmost_subplotspec())

    # Check unexpected kwargs
    with pytest.raises(TypeError):
        make_1D_axes(paramnames, foo='bar')


def test_make_2D_axes():
    paramnames_x = ['A', 'B', 'C']
    paramnames_y = ['B', 'A', 'D']
    tex = {'A':'tA', 'B':'tB', 'C':'tC', 'D':'tD', 'E':'tE'}

    # 2D axes
    fig, axes = make_2D_axes(paramnames_x, paramnames_y)
    assert(isinstance(fig, Figure))
    assert(isinstance(axes, DataFrame))
    assert_array_equal(axes.index, paramnames_y)
    assert_array_equal(axes.columns, paramnames_x)

    # Axes labels
    for p, ax in axes.iloc[:,0].iteritems():
        assert(ax.get_ylabel() == p)

    for p, ax in axes.iloc[-1].iteritems():
        assert(ax.get_xlabel() == p)

    for ax in axes.iloc[:-1,1:].values.flatten():
        assert(ax.get_xlabel()=='')
        assert(ax.get_ylabel()=='')
    plt.close(fig)

    # Check fig argument
    fig0 = plt.figure()
    fig1 = plt.figure()
    fig, axes = make_2D_axes(paramnames_x)
    assert(fig is fig1)
    fig, axes = make_2D_axes(paramnames_x, fig=fig0)
    assert(fig is fig0)
    plt.close(fig0)
    plt.close(fig1)

    # Check gridspec argument
    grid = gs.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[3,1])
    g00 = grid[0,0]
    fig, axes = make_2D_axes(paramnames_x, subplot_spec=g00)
    assert(g00 is axes.iloc[0,0].get_subplotspec().get_topmost_subplotspec())
    plt.close(fig)

    # Check unexpected kwargs
    with pytest.raises(TypeError):
        make_2D_axes(paramnames_x, foo='bar')

def test_plot_1d():
    fig, ax = plt.subplots()
    data = numpy.random.randn(1000)

    # Check height
    line, = plot_1d(ax, data)
    assert(isinstance(line, Line2D))
    assert(line.get_ydata().max() <=1)

    # Check arguments are passed onward to underlying function
    line, = plot_1d(ax, data, color='r')
    assert(line.get_color()=='r')

    # Check xmin
    xmin = -0.5
    line, = plot_1d(ax, data, xmin=xmin)
    assert((line.get_xdata()>=xmin).all())

    # Check xmax
    xmax = 0.5
    line, = plot_1d(ax, data, xmax=xmax)
    assert((line.get_xdata()<=xmax).all())

    # Check xmin and xmax
    line, = plot_1d(ax, data, xmin=xmin, xmax=xmax)
    assert((line.get_xdata()<=xmax).all())
    assert((line.get_xdata()>=xmin).all())


def test_contour_plot_2d():
    ax = plt.gca()
    data_x = numpy.random.randn(1000)
    data_y = numpy.random.randn(1000)
    c = contour_plot_2d(ax, data_x, data_y)
    assert(isinstance(c, QuadContourSet))

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


def test_scatter_plot_2d():
    fig, ax = plt.subplots()
    data_x = numpy.random.randn(1000)
    data_y = numpy.random.randn(1000)
    lines, = scatter_plot_2d(ax, data_x, data_y)
    assert(isinstance(lines, Line2D))

    xmin, xmax, ymin, ymax = -0.5, 0.5, -0.5, 0.5
    ax = plt.gca()
    contour_plot_2d(ax, data_x, data_y, xmin=xmin)
    assert(ax.get_xlim()[0] >= xmin)
    plt.close()

    ax = plt.gca()
    contour_plot_2d(ax, data_x, data_y, xmax=xmax)
    assert(ax.get_xlim()[1] <= xmax)
    plt.close()

    ax = plt.gca()
    contour_plot_2d(ax, data_x, data_y, ymin=ymin)
    assert(ax.get_ylim()[0] >= ymin)
    plt.close()

    ax = plt.gca()
    contour_plot_2d(ax, data_x, data_y, ymax=ymax)
    assert(ax.get_ylim()[1] <= ymax)
    plt.close()
