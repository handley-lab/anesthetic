"""Lower-level plotting tools.

Routines that may be of use to users wishing for more fine-grained control may
wish to use.

- ``make_1d_axes``
- ``make_2d_axes``
- ``get_legend_proxy``

to create a set of axes and legend proxies.

"""
import numpy
import pandas
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GS, GridSpecFromSubplotSpec as SGS
from anesthetic.kde import kde_1d, kde_2d
from anesthetic.utils import check_bounds, nest_level, unique
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PathCollection


def make_1d_axes(params, **kwargs):
    """Create a set of axes for plotting 1D marginalised posteriors.

    Parameters
    ----------
        params: list(str)
            names of parameters.

        tex: dict(str:str), optional
            Dictionary mapping params to tex plot labels.

        fig: matplotlib.figure.Figure, optional
            Figure to plot on.
            Default: matplotlib.pyplot.figure()

        ncols: int
            Number of columns in the plot
            option, default ceil(sqrt(num_params))

        subplot_spec: matplotlib.gridspec.GridSpec, optional
            gridspec to plot array as part of a subfigure
            Default: None

    Returns
    -------
    fig: matplotlib.figure.Figure
        New or original (if supplied) figure object

    axes: pandas.Series(matplotlib.axes.Axes)
        Pandas array of axes objects

    """
    axes = pandas.Series(index=numpy.atleast_1d(params), dtype=object)
    axes[:] = None
    tex = kwargs.pop('tex', {})
    fig = kwargs.pop('fig') if 'fig' in kwargs else plt.figure()
    ncols = kwargs.pop('ncols', int(numpy.ceil(numpy.sqrt(len(axes)))))
    nrows = int(numpy.ceil(len(axes)/float(ncols)))
    if 'subplot_spec' in kwargs:
        grid = SGS(nrows, ncols, wspace=0,
                   subplot_spec=kwargs.pop('subplot_spec'))
    else:
        grid = GS(nrows, ncols, wspace=0)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    tex = {p: tex[p] if p in tex else p for p in axes.index}

    for p, g in zip(axes.index, grid):
        axes[p] = ax = fig.add_subplot(g)
        ax.set_xlabel(tex[p])
        ax.set_yticks([])

    for x, ax in axes.dropna().iteritems():
        ax.xaxis.set_major_locator(MaxNLocator(2, integer=True))

    return fig, axes


def make_2d_axes(params, **kwargs):
    """Create a set of axes for plotting 2D marginalised posteriors.

    Parameters
    ----------
        params: lists of parameters
            Can be either:
            * list(str) if the x and y axes are the same
            * [list(str),list(str)] if the x and y axes are different
            Strings indicate the names of the parameters

        tex: dict(str:str), optional
            Dictionary mapping params to tex plot labels.
            Default: params

        upper: None or logical, optional
            Whether to create plots in the upper triangle.
            If None do both. Default: None

        diagonal: True, optional
            Whether to create 1D marginalised plots on the diagonal.
            Default: True

        fig: matplotlib.figure.Figure, optional
            Figure to plot on.
            Default: matplotlib.pyplot.figure()

        subplot_spec: matplotlib.gridspec.GridSpec, optional
            gridspec to plot array as part of a subfigure.
            Default: None

    Returns
    -------
    fig: matplotlib.figure.Figure
        New or original (if supplied) figure object

    axes: pandas.DataFrame(matplotlib.axes.Axes)
        Pandas array of axes objects

    """
    if nest_level(params) == 2:
        xparams, yparams = params
    else:
        xparams = yparams = params
    axes = pandas.DataFrame(index=numpy.atleast_1d(yparams),
                            columns=numpy.atleast_1d(xparams),
                            dtype=object)
    axes[:][:] = None

    tex = kwargs.pop('tex', {})
    fig = kwargs.pop('fig') if 'fig' in kwargs else plt.figure()
    if 'subplot_spec' in kwargs:
        grid = SGS(*axes.shape, hspace=0, wspace=0,
                   subplot_spec=kwargs.pop('subplot_spec'))
    else:
        grid = GS(*axes.shape, hspace=0, wspace=0)

    upper = kwargs.pop('upper', None)
    diagonal = kwargs.pop('diagonal', True)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if axes.size == 0:
        return fig, axes

    tex = {p: tex[p] if p in tex else p
           for p in numpy.concatenate((axes.index, axes.columns))}

    all_params = list(axes.index) + list(axes.columns)

    for j, y in enumerate(axes.index):
        for i, x in enumerate(axes.columns):
            lower = not (x in axes.index and y in axes.columns
                         and all_params.index(x) > all_params.index(y))

            if x == y and not diagonal:
                continue

            if upper == lower and x != y:
                continue

            if axes[x][y] is None:
                sx = list(axes[x].dropna())
                sx = sx[0] if sx else None
                sy = list(axes.T[y].dropna())
                sy = sy[0] if sy else None
                axes[x][y] = fig.add_subplot(grid[j, i],
                                             sharex=sx, sharey=sy)
            if x == y:
                axes[x][y].twin = axes[x][y].twinx()
                axes[x][y].twin.set_yticks([])
                axes[x][y].twin.set_ylim(0, 1.1)

            axes[x][y]._upper = not lower

    for y, ax in axes.bfill(axis=1).iloc[:, 0].dropna().iteritems():
        ax.set_ylabel(tex[y])

    for x, ax in axes.ffill(axis=0).iloc[-1, :].dropna().iteritems():
        ax.set_xlabel(tex[x])

    for y, ax in axes.iterrows():
        ax_ = ax.dropna()
        if len(ax_):
            for a in ax_[1:]:
                a.tick_params('y', left=False, labelleft=False)

    for x, ax in axes.iteritems():
        ax_ = ax.dropna()
        if len(ax_):
            for a in ax_[:-1]:
                a.tick_params('x', bottom=False, labelbottom=False)

    for y, ax in axes.bfill(axis=1).iloc[:, 0].dropna().iteritems():
        ax.yaxis.set_major_locator(MaxNLocator(3, prune='both'))

    for x, ax in axes.ffill(axis=0).iloc[-1, :].dropna().iteritems():
        ax.xaxis.set_major_locator(MaxNLocator(3, prune='both'))

    return fig, axes


def plot_1d(ax, data, *args, **kwargs):
    """Plot a 1d marginalised distribution.

    This functions as a wrapper around matplotlib.axes.Axes.plot, with a kernel
    density estimation computation in between. All remaining keyword arguments
    are passed onwards.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axis object to plot on

    data: numpy.array
        Uniformly weighted samples to generate kernel density estimator.

    xmin, xmax: float
        lower/upper prior bound
        optional, default None

    Returns
    -------
    lines: matplotlib.lines.Line2D
        A list of line objects representing the plotted data (same as
        matplotlib matplotlib.axes.Axes.plot command)

    """
    if data.max()-data.min() <= 0:
        return

    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)

    x, p = kde_1d(data, xmin, xmax)
    p /= p.max()
    i = (p >= 1e-2)

    ans = ax.plot(x[i], p[i], *args, **kwargs)
    ax.set_xlim(*check_bounds(x[i], xmin, xmax), auto=True)
    return ans


def contour_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot a 2d marginalised distribution as contours.

    This functions as a wrapper around matplotlib.axes.Axes.contour, and
    matplotlib.axes.Axes.contourf with a kernel density estimation computation
    in between. All remaining keyword arguments are passed onwards to both
    functions.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axis object to plot on

    data_x, data_y: numpy.array
        x and y coordinates of uniformly weighted samples to generate kernel
        density estimator.

    xmin, xmax, ymin, ymax: float
        lower/upper prior bounds in x/y coordinates
        optional, default None

    Returns
    -------
    c: matplotlib.contour.QuadContourSet
        A set of contourlines or filled regions

    """
    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)
    ymin = kwargs.pop('ymin', None)
    ymax = kwargs.pop('ymax', None)
    color = kwargs.pop('color', next(ax._get_lines.prop_cycler)['color'])

    x, y, pdf = kde_2d(data_x, data_y,
                       xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    pdf /= pdf.max()
    p = sorted(pdf.flatten())
    m = numpy.cumsum(p)
    m /= m[-1]
    interp = interp1d([0]+list(m)+[1], [0]+list(p)+[1])
    contours = list(interp([0.05, 0.33]))+[1]

    # Correct non-zero edges
    if min(p) != 0:
        contours = [min(p)] + contours

    # Correct level sets
    for i in range(1, len(contours)):
        if contours[i-1] == contours[i]:
            for j in range(i):
                contours[j] = contours[j] - 1e-5

    i = (pdf >= 1e-2).any(axis=0)
    j = (pdf >= 1e-2).any(axis=1)

    cmap = basic_cmap(color)
    zorder = max([child.zorder for child in ax.get_children()])

    cbar = ax.contourf(x[i], y[j], pdf[numpy.ix_(j, i)], contours,
                       vmin=0, vmax=1.0, cmap=cmap, zorder=zorder+1,
                       *args, **kwargs)
    for c in cbar.collections:
        c.set_cmap(cmap)

    ax.contour(x[i], y[j], pdf[numpy.ix_(j, i)], contours,
               vmin=0, vmax=1.2, linewidths=0.5, colors='k', zorder=zorder+2,
               *args, **kwargs)
    ax.set_xlim(*check_bounds(x[i], xmin, xmax), auto=True)
    ax.set_ylim(*check_bounds(y[j], ymin, ymax), auto=True)
    return cbar


def scatter_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot samples from a 2d marginalised distribution.

    This functions as a wrapper around matplotlib.axes.Axes.plot, enforcing any
    prior bounds. All remaining keyword arguments are passed onwards.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axis object to plot on

    data_x, data_y: numpy.array
        x and y coordinates of uniformly weighted samples to generate kernel
        density estimator.

    xmin, xmax, ymin, ymax: float
        lower/upper prior bounds in x/y coordinates
        optional, default None

    Returns
    -------
    lines: matplotlib.lines.Line2D
        A list of line objects representing the plotted data (same as
        matplotlib matplotlib.axes.Axes.plot command)

    """
    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)
    ymin = kwargs.pop('ymin', None)
    ymax = kwargs.pop('ymax', None)

    points = ax.plot(data_x, data_y, 'o', markersize=1, *args, **kwargs)
    ax.set_xlim(*check_bounds(data_x, xmin, xmax), auto=True)
    ax.set_ylim(*check_bounds(data_y, ymin, ymax), auto=True)
    return points


def get_legend_proxy(fig):
    """Extract a proxy for plotting onto a legend.

    Example usage:
        >>> fig, axes = modelA.plot_2d()
        >>> modelB.plot_2d(axes)
        >>> proxy = get_legend_proxy(fig)
        >>> fig.legend(proxy, ['A', 'B']

    Parameters
    ----------
        fig: matplotlib.figure.Figure
            Figure to extract colors from.

    """
    cmaps = [coll.get_cmap() for ax in fig.axes for coll in ax.collections
             if isinstance(coll, PathCollection)]
    cmaps = unique(cmaps)

    if not cmaps:
        colors = [line.get_color() for ax in fig.axes for line in ax.lines]
        colors = unique(colors)
        cmaps = [basic_cmap(color) for color in colors]

    proxy = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap(0.999),
                           edgecolor=cmap(0.33), linewidth=2)
             for cmap in cmaps]

    return proxy


def basic_cmap(color):
    """Construct basic colormap a single color."""
    return LinearSegmentedColormap.from_list(color, ['#ffffff', color])
