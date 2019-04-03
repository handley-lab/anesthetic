import numpy
import pandas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from fastkde import fastKDE
from anesthetic.kde import kde_1d, kde_2d
from anesthetic.utils import check_bounds
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
import matplotlib.colors

def make_1D_axes(paramnames, **kwargs):
    """ Create a set of axes for plotting 1D marginalised posteriors

    Parameters
    ----------
        paramnames: list(str)
            names of parameters.

        tex: dict(str:str)
            Dictionary mapping paramnames to tex plot labels.
            optional, default paramnames

        fig: matplotlib.figure.Figure
            Figure to plot on
            optional, default last figure matplotlib.pyplot.gcf()

        ncols: int
            Number of columns in the plot
            option, default ceil(sqrt(num_params))
            
        subplot_spec: matplotlib.gridspec.GridSpec
            gridspec to plot array as part of a subfigure
            optional, default None

    Returns
    -------
    fig: matplotlib.figure.Figure
        New or original (if supplied) figure object

    axes: pandas.DataFrame(matplotlib.axes.Axes)
        Pandas array of axes objects 
    """

    tex = kwargs.pop('tex', {})
    fig = kwargs.pop('fig', plt.gcf())
    ncols = kwargs.pop('ncols', int(numpy.ceil(numpy.sqrt(len(paramnames)))))
    nrows = int(numpy.ceil(len(paramnames)/ncols))
    if 'subplot_spec' in kwargs:
        grid = gs.GridSpecFromSubplotSpec(nrows, ncols, wspace=0,
                                          subplot_spec=kwargs.pop('subplot_spec'))
    else:
        grid = gs.GridSpec(nrows, ncols, wspace=0)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    tex = {p:tex[p] if p in tex else p for p in paramnames}
    axes = pandas.Series(index=paramnames, dtype=object)

    for p, g in zip(paramnames, grid):
        axes[p] = ax = fig.add_subplot(g) 
        ax.set_xlabel(tex[p])
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(2))

    return fig, axes


def make_2D_axes(paramnames, paramnames_y=None, **kwargs):
    """ Create a set of axes for plotting 2D marginalised posteriors

    Parameters
    ----------
        paramnames: list(str)
            names of parameters.

        paramnames_y: list(str)
            names of parameters.
            optional, default paramnames

        tex: dict(str:str)
            Dictionary mapping paramnames to tex plot labels.
            optional, default paramnames

        fig: matplotlib.figure.Figure
            Figure to plot on
            optional, default last figure matplotlib.pyplot.gcf()
            
        subplot_spec: matplotlib.gridspec.GridSpec
            gridspec to plot array as part of a subfigure
            optional, default None

    Returns
    -------
    fig: matplotlib.figure.Figure
        New or original (if supplied) figure object

    axes: pandas.DataFrame(matplotlib.axes.Axes)
        Pandas array of axes objects 
    """
    paramnames_x = paramnames
    if paramnames_y is None:
        paramnames_y = paramnames
    tex = kwargs.pop('tex', {})
    fig = kwargs.pop('fig', plt.gcf())
    nx = len(paramnames_x)
    ny = len(paramnames_y)
    if 'subplot_spec' in kwargs:
        grid = gs.GridSpecFromSubplotSpec(ny, nx, hspace=0, wspace=0,
                                          subplot_spec=kwargs.pop('subplot_spec'))
    else:
        grid = gs.GridSpec(ny, nx, hspace=0, wspace=0)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    tex = {p:tex[p] if p in tex else p for p in numpy.concatenate((paramnames_x, paramnames_y))}
    axes = pandas.DataFrame(index=paramnames_y, columns=paramnames_x, dtype=object)
    axes[:][:] = None


    # 2D plots
    for y, py in enumerate(paramnames_y):
        for x, px in enumerate(paramnames_x):
            sx = axes[px][paramnames_y[0]]
            sy = axes[paramnames_y[0]][py]
            axes[px][py] = ax = fig.add_subplot(grid[y,x], sharex=sx, sharey=sy)
            
            ax.label_outer()

            if y == ny-1:
                ax.set_xlabel(tex[px])
                ax.xaxis.set_major_locator(MaxNLocator(2))
            else:
                ax.tick_params('x',bottom=False)

            if x == 0:
                ax.set_ylabel(tex[py])
                ax.yaxis.set_major_locator(MaxNLocator(2))
            else:
                ax.tick_params('y',left=False)

    return fig, axes


def plot_1d(ax, data, *args, **kwargs):
    """Plot a 1d marginalised distribution

    This functions as a wrapper around matplotlib.axes.Axes.plot, with a kernel
    density estimation computation in between. All remaining keyword arguments
    are passed onwards.

    To avoid intefering with y-axis sharing, one-dimensional plots are created
    on a separate axis, which is monkey-patched onto the argument ax as the
    attribute ax.twin.

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

    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)

    if not hasattr(ax, 'twin'):
        ax.twin = ax.twinx() 
        ax.twin.set_yticks([])
        ax.twin.set_ylim(0,1.1)
        if not ax.is_first_col():
            ax.tick_params('y',left=False)

    x, p = kde_1d(data, xmin, xmax)
    p /= p.max()
    i = (p>=1e-2)

    ans = ax.twin.plot(x[i], p[i], *args, **kwargs)
    ax.twin.set_xlim(*check_bounds(x[i], xmin, xmax), auto=True)
    return ans


def contour_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot a 2d marginalised distribution as contours

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

    x, y, pdf = kde_2d(data_x, data_y, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    pdf /= pdf.max()
    p = sorted(pdf.flatten())
    m = numpy.cumsum(p)
    m /= m[-1]
    interp = interp1d([0]+list(m)+[1],[0]+list(p)+[1])
    contours = list(interp([0.05, 0.33]))+[1]

    # Correct non-zero edges
    if min(p) != 0:
        contours = [min(p)] + contours

    # Correct level sets
    for i in range(1, len(contours)):
        if contours[i-1]==contours[i]:
            for j in range(i):
                contours[j] = contours[j] - 1e-5
            
    i = (pdf>=1e-2).any(axis=0)
    j = (pdf>=1e-2).any(axis=1)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(color, ['#ffffff', color])

    cbar = ax.contourf(x[i], y[j], pdf[numpy.ix_(j,i)], contours, vmin=0, vmax=1.0, cmap=cmap, *args, **kwargs)  
    ax.contour(x[i], y[j], pdf[numpy.ix_(j,i)], contours, vmin=0,vmax=1.2, linewidths=0.5, colors='k', *args, **kwargs)  
    ax.set_xlim(*check_bounds(x[i], xmin, xmax), auto=True)
    ax.set_ylim(*check_bounds(y[j], ymin, ymax), auto=True)
    return cbar


def scatter_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot samples from a 2d marginalised distribution

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
