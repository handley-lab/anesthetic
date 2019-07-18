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
try:
    from astropy.visualization import hist
except ImportError:
    from matplotlib.pyplot import hist
from anesthetic.kde import kde_1d, kde_2d
from anesthetic.utils import check_bounds, nest_level, unique
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D


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

        upper, lower, diagonal: logical, optional
            Whether to create 2D marginalised plots above or below the
            diagonal, or to create a 1D marginalised plot on the diagonal.
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

    upper = kwargs.pop('upper', True)
    lower = kwargs.pop('lower', True)
    diagonal = kwargs.pop('diagonal', True)

    axes = pandas.DataFrame(index=numpy.atleast_1d(yparams),
                            columns=numpy.atleast_1d(xparams),
                            dtype=object)
    axes[:][:] = None
    all_params = list(axes.columns) + list(axes.index)

    for j, y in enumerate(axes.index):
        for i, x in enumerate(axes.columns):
            if all_params.index(x) < all_params.index(y):
                if lower:
                    axes[x][y] = -1
            elif all_params.index(x) > all_params.index(y):
                if upper:
                    axes[x][y] = +1
            elif diagonal:
                axes[x][y] = 0

    axes.dropna(axis=0, how='all', inplace=True)
    axes.dropna(axis=1, how='all', inplace=True)

    tex = kwargs.pop('tex', {})
    tex = {p: tex[p] if p in tex else p for p in all_params}
    fig = kwargs.pop('fig') if 'fig' in kwargs else plt.figure()
    if 'subplot_spec' in kwargs:
        grid = SGS(*axes.shape, hspace=0, wspace=0,
                   subplot_spec=kwargs.pop('subplot_spec'))
    else:
        grid = GS(*axes.shape, hspace=0, wspace=0)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if axes.size == 0:
        return fig, axes
    position = axes.copy()
    axes[:][:] = None
    for j, y in enumerate(axes.index):
        for i, x in enumerate(axes.columns):
            if position[x][y] is not None:
                sx = list(axes[x].dropna())
                sx = sx[0] if sx else None
                sy = list(axes.T[y].dropna())
                sy = sy[0] if sy else None
                axes[x][y] = fig.add_subplot(grid[j, i],
                                             sharex=sx, sharey=sy)

                if position[x][y] == 0:
                    axes[x][y].twin = axes[x][y].twinx()
                    axes[x][y].twin.set_yticks([])
                    axes[x][y].twin.set_ylim(0, 1.1)
                    axes[x][y].set_zorder(axes[x][y].twin.get_zorder() + 1)
                    axes[x][y].lines = axes[x][y].twin.lines
                    axes[x][y].patches = axes[x][y].twin.patches
                    axes[x][y].collections = axes[x][y].twin.collections
                    axes[x][y].containers = axes[x][y].twin.containers
                    axes[x][y].position = 'diagonal'
                elif position[x][y] == 1:
                    axes[x][y].position = 'upper'
                elif position[x][y] == -1:
                    axes[x][y].position = 'lower'

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
    if len(data) == 0:
        return numpy.zeros(0), numpy.zeros(0)

    if max(data)-min(data) <= 0:
        return

    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)

    x, p = kde_1d(data, xmin, xmax)
    p /= p.max()
    i = (p >= 1e-2)

    ans = ax.plot(x[i], p[i], *args, **kwargs)
    ax.set_xlim(*check_bounds(x[i], xmin, xmax), auto=True)
    return ans


def hist_1d(ax, data, *args, **kwargs):
    """Plot a 1d histogram.

    This functions is a wrapper around matplotlib.axes.Axes.hist, or
    astropy.visualization.hist if it is available. astropy's hist function
    allows for a more sophisticated calculation of the bins. All remaining
    keyword arguments are passed onwards.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axis object to plot on

    data: numpy.array
        Uniformly weighted samples to generate kernel density estimator.

    xmin, xmax: float
        lower/upper prior bound
        optional, default data.min() and data.max()
        cannot be None (reverts to default in that case)

    Returns
    -------
    patches : list or list of lists
        Silent list of individual patches used to create the histogram
        or list of such list if multiple input datasets.

    Other Parameters
    ----------------
    **kwargs : `~matplotlib.axes.Axes.hist` properties

    """
    if data.max()-data.min() <= 0:
        return

    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)
    if xmin is None:
        xmin = data.min()
    if xmax is None:
        xmax = data.max()
    histtype = kwargs.pop('histtype', 'bar')

    plt.sca(ax=ax)
    h, edges, bars = hist(data, range=(xmin, xmax), histtype=histtype,
                          *args, **kwargs)
    # As the y-axis on the diagonal 1D plots of the triangle plot won't
    # be labelled, we set the maximum bar height to 1:
    if histtype == 'bar':
        for b in bars:
            b.set_height(b.get_height() / h.max())
    elif histtype == 'step' or histtype == 'stepfilled':
        trans = Affine2D().scale(sx=1, sy=1./h.max()) + ax.transData
        bars[0].set_transform(trans)
    ax.set_xlim(*check_bounds(edges, xmin, xmax), auto=True)
    ax.set_ylim(0, 1.1)
    return bars


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
    label = kwargs.pop('label', None)
    zorder = kwargs.pop('zorder', 1)
    color = kwargs.pop('color', next(ax._get_lines.prop_cycler)['color'])
    if len(data_x) == 0 or len(data_y) == 0:
        return numpy.zeros(0), numpy.zeros(0), numpy.zeros((0, 0))

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

    cbar = ax.contourf(x[i], y[j], pdf[numpy.ix_(j, i)], contours, cmap=cmap,
                       zorder=zorder, vmin=0, vmax=1.0, *args, **kwargs)
    for c in cbar.collections:
        c.set_cmap(cmap)

    ax.contour(x[i], y[j], pdf[numpy.ix_(j, i)], contours, zorder=zorder,
               vmin=0, vmax=1.2, linewidths=0.5, colors='k', *args, **kwargs)
    ax.patches += [plt.Rectangle((0, 0), 1, 1, fc=cmap(0.999), ec=cmap(0.33),
                                 lw=2, label=label)]

    ax.set_xlim(*check_bounds(x[i], xmin, xmax), auto=True)
    ax.set_ylim(*check_bounds(y[j], ymin, ymax), auto=True)
    return cbar


def scatter_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot samples from a 2d marginalised distribution.

    This functions as a wrapper around matplotlib.axes.Axes.scatter, enforcing
    any prior bounds. All remaining keyword arguments are passed onwards.

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
    matplotlib.collections.PathCollection object
        A PathCollection object representing the plotted data (same as
        matplotlib matplotlib.axes.Axes.scatter command)

    """
    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)
    ymin = kwargs.pop('ymin', None)
    ymax = kwargs.pop('ymax', None)
    s = kwargs.pop('s', 3)

    points = ax.scatter(data_x, data_y, s=s, *args, **kwargs)
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
    from warnings import warn
    warn("`get_legend_proxy` is deprecated and might be deleted in the "
         "future. You can now simply use `axes[x][y].legend()` or do "
         "`handles, labels = axes[x][y].get_legend_handles_labels()` "
         "and pass the handles and labels to the figure legend "
         "`fig.legend(handles, labels)`.", FutureWarning)
    cmaps = [coll.get_cmap() for ax in fig.axes for coll in ax.collections
             if isinstance(coll.get_cmap(), LinearSegmentedColormap)]
    cmaps = unique(cmaps)

    if not cmaps:
        colors = [line.get_color() for ax in fig.axes for line in ax.lines]
        colors = unique(colors)
        cmaps = [basic_cmap(color) for color in colors]

    proxy = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap(0.999),
                           edgecolor=cmap(0.33), linewidth=2)
             for cmap in cmaps]

    if not cmaps:
        colors = [coll.get_ec()[0]
                  for ax in fig.axes
                  for coll in ax.collections
                  if isinstance(coll, LineCollection)]
        colors = numpy.unique(colors, axis=0)
        cmaps = [basic_cmap(color) for color in colors]
        proxy = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap(0.0),
                               edgecolor=cmap(0.999), linewidth=1)
                 for cmap in cmaps]

    return proxy


def basic_cmap(color):
    """Construct basic colormap a single color."""
    return LinearSegmentedColormap.from_list(color, ['#ffffff', color])
