"""Lower-level plotting tools.

Routines that may be of use to users wishing for more fine-grained control may
wish to use.

- ``make_1d_axes``
- ``make_2d_axes``

to create a set of axes and legend proxies.

"""
import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec as GS, GridSpecFromSubplotSpec as SGS
try:
    from astropy.visualization import hist
except ImportError:
    pass
try:
    from anesthetic.kde import fastkde_1d, fastkde_2d
except ImportError:
    pass
import matplotlib.cbook as cbook
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Affine2D
from anesthetic.utils import check_bounds, nest_level
from anesthetic.utils import (sample_compression_1d, quantile,
                              triangular_sample_compression_2d,
                              iso_probability_contours,
                              iso_probability_contours_from_samples,
                              scaled_triangulation, match_contour_to_contourf)
from anesthetic.boundary import cut_and_normalise_gaussian


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
    axes = pandas.Series(index=np.atleast_1d(params), dtype=object)
    axes[:] = None
    tex = kwargs.pop('tex', {})
    fig = kwargs.pop('fig') if 'fig' in kwargs else plt.figure()
    ncols = kwargs.pop('ncols', int(np.ceil(np.sqrt(len(axes)))))
    nrows = int(np.ceil(len(axes)/float(ncols)))
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

    axes = pandas.DataFrame(index=np.atleast_1d(yparams),
                            columns=np.atleast_1d(xparams),
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
    spec = kwargs.pop('subplot_spec', None)
    if axes.shape[0] != 0 and axes.shape[1] != 0:
        if spec is not None:
            grid = SGS(*axes.shape, hspace=0, wspace=0, subplot_spec=spec)
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
                    make_diagonal(axes[x][y])
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


def fastkde_plot_1d(ax, data, *args, **kwargs):
    """Plot a 1d marginalised distribution.

    This functions as a wrapper around matplotlib.axes.Axes.plot, with a kernel
    density estimation computation provided by the package fastkde in between.
    All remaining keyword arguments are passed onwards.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axis object to plot on

    data: np.array
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
        return np.zeros(0), np.zeros(0)

    if data.max()-data.min() <= 0:
        return

    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)
    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', (next(ax._get_lines.prop_cycler)['color']
                                 if cmap is None else cmap(0.68)))
    q = kwargs.pop('q', '5sigma')
    q = quantile_plot_interval(q=q)

    try:
        x, p = fastkde_1d(data, xmin, xmax)
    except NameError:
        raise ImportError("You need to install fastkde to use fastkde")
    p /= p.max()
    i = ((x > quantile(x, q[0], p)) & (x < quantile(x, q[1], p)))

    ans = ax.plot(x[i], p[i], color=color, *args, **kwargs)
    ax.set_xlim(*check_bounds(x[i], xmin, xmax), auto=True)
    return ans


def kde_plot_1d(ax, data, *args, **kwargs):
    """Plot a 1d marginalised distribution.

    This functions as a wrapper around matplotlib.axes.Axes.plot, with a kernel
    density estimation computation provided by scipy.stats.gaussian_kde in
    between. All remaining keyword arguments are passed onwards.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axis object to plot on.

    data: np.array
        Samples to generate kernel density estimator.

    weights: np.array, optional
        Sample weights.

    ncompress: int, optional
        Degree of compression. Default 1000

    xmin, xmax: float
        lower/upper prior bound.
        optional, default None

    Returns
    -------
    lines: matplotlib.lines.Line2D
        A list of line objects representing the plotted data (same as
        matplotlib matplotlib.axes.Axes.plot command)

    """
    if len(data) == 0:
        return np.zeros(0), np.zeros(0)

    if data.max()-data.min() <= 0:
        return

    kwargs = normalize_kwargs(
        kwargs,
        dict(linewidth=['lw'], linestyle=['ls'], color=['c']),
        drop=['fc', 'ec'])
    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)
    weights = kwargs.pop('weights', None)
    ncompress = kwargs.pop('ncompress', 1000)
    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', (next(ax._get_lines.prop_cycler)['color']
                                 if cmap is None else cmap(0.68)))
    q = kwargs.pop('q', '5sigma')
    q = quantile_plot_interval(q=q)

    if weights is not None:
        data = data[weights != 0]
        weights = weights[weights != 0]

    x, w = sample_compression_1d(data, weights, ncompress)
    kde = gaussian_kde(x, weights=w)
    p = kde(x)
    p /= p.max()
    i = ((x > quantile(x, q[0], w)) & (x < quantile(x, q[1], w)))
    if xmin is not None:
        i = i & (x > xmin)
    if xmax is not None:
        i = i & (x < xmax)
    sigma = np.sqrt(kde.covariance[0, 0])
    pp = cut_and_normalise_gaussian(x[i], p[i], sigma, xmin, xmax)
    pp /= pp.max()
    ans = ax.plot(x[i], pp, color=color, *args, **kwargs)
    ax.set_xlim(*check_bounds(x[i], xmin, xmax), auto=True)
    return ans


def hist_plot_1d(ax, data, *args, **kwargs):
    """Plot a 1d histogram.

    This functions is a wrapper around matplotlib.axes.Axes.hist. All remaining
    keyword arguments are passed onwards.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axis object to plot on

    data: np.array
        Samples to generate histogram from

    weights: np.array, optional
        Sample weights.

    xmin, xmax: float
        lower/upper prior bound.
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
    plotter = kwargs.pop('plotter', '')
    weights = kwargs.pop('weights', None)
    if xmin is None or not np.isfinite(xmin):
        xmin = quantile(data, 0.01, weights)
    if xmax is None or not np.isfinite(xmax):
        xmax = quantile(data, 0.99, weights)
    histtype = kwargs.pop('histtype', 'bar')
    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', (next(ax._get_lines.prop_cycler)['color']
                                 if cmap is None else cmap(0.68)))

    if plotter == 'astropyhist':
        try:
            h, edges, bars = hist(data, ax=ax, color=color, range=(xmin, xmax),
                                  histtype=histtype, *args, **kwargs)
        except NameError:
            raise ImportError("You need to install astropy to use astropyhist")
    else:
        h, edges, bars = ax.hist(data, color=color, range=(xmin, xmax),
                                 histtype=histtype, weights=weights,
                                 *args, **kwargs)

    if histtype == 'bar':
        for b in bars:
            b.set_height(b.get_height() / h.max())
    elif histtype == 'step' or histtype == 'stepfilled':
        trans = Affine2D().scale(sx=1, sy=1./h.max()) + ax.transData
        bars[0].set_transform(trans)

    ax.set_xlim(*check_bounds(edges, xmin, xmax), auto=True)
    ax.set_ylim(0, 1.1)
    return bars


def fastkde_contour_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot a 2d marginalised distribution as contours.

    This functions as a wrapper around matplotlib.axes.Axes.contour, and
    matplotlib.axes.Axes.contourf with a kernel density estimation computation
    in between. All remaining keyword arguments are passed onwards to both
    functions.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axis object to plot on

    data_x, data_y: np.array
        x and y coordinates of uniformly weighted samples to generate kernel
        density estimator.

    levels: list
        amount of mass within each iso-probability contour.
        optional, default [0.68, 0.95]

    xmin, xmax, ymin, ymax: float
        lower/upper prior bounds in x/y coordinates
        optional, default None

    Returns
    -------
    c: matplotlib.contour.QuadContourSet
        A set of contourlines or filled regions

    """
    kwargs = normalize_kwargs(kwargs, dict(linewidths=['linewidth', 'lw'],
                                           linestyles=['linestyle', 'ls'],
                                           color=['c'],
                                           facecolor=['fc'],
                                           edgecolor=['ec']))
    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)
    ymin = kwargs.pop('ymin', None)
    ymax = kwargs.pop('ymax', None)
    label = kwargs.pop('label', None)
    zorder = kwargs.pop('zorder', 1)
    levels = kwargs.pop('levels', [0.68, 0.95])
    color = kwargs.pop('color', next(ax._get_lines.prop_cycler)['color'])
    facecolor = kwargs.pop('facecolor', color)
    edgecolor = kwargs.pop(
        'edgecolor', color if facecolor in [None, 'None', 'none'] else 'k')
    kwargs.pop('q', None)

    if len(data_x) == 0 or len(data_y) == 0:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0))

    try:
        x, y, pdf = fastkde_2d(data_x, data_y,
                               xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    except NameError:
        raise ImportError("You need to install fastkde to use fastkde")

    levels = iso_probability_contours(pdf, contours=levels)

    i = (pdf >= levels[0]*0.5).any(axis=0)
    j = (pdf >= levels[0]*0.5).any(axis=1)

    if facecolor not in [None, 'None', 'none']:
        linewidths = kwargs.pop('linewidths', 0.5)
        cmap = kwargs.pop('cmap', basic_cmap(facecolor))
        contf = ax.contourf(x[i], y[j], pdf[np.ix_(j, i)], levels, cmap=cmap,
                            zorder=zorder, vmin=0, vmax=pdf.max(),
                            *args, **kwargs)
        for c in contf.collections:
            c.set_cmap(cmap)
        ax.patches += [plt.Rectangle((0, 0), 0, 0, lw=2, label=label,
                                     fc=cmap(0.999), ec=cmap(0.32))]
        cmap = None
    else:
        linewidths = kwargs.pop('linewidths',
                                plt.rcParams.get('lines.linewidth'))
        cmap = kwargs.pop('cmap', None)
        contf = None
        ax.patches += [
            plt.Rectangle((0, 0), 0, 0, lw=2, label=label,
                          fc='None' if cmap is None else cmap(0.999),
                          ec=edgecolor if cmap is None else cmap(0.32))
        ]
        edgecolor = edgecolor if cmap is None else None

    vmin, vmax = match_contour_to_contourf(levels, vmin=0, vmax=pdf.max())
    cont = ax.contour(x[i], y[j], pdf[np.ix_(j, i)], levels, zorder=zorder,
                      vmin=vmin, vmax=vmax, linewidths=linewidths,
                      colors=edgecolor, cmap=cmap, *args, **kwargs)

    ax.set_xlim(*check_bounds(x[i], xmin, xmax), auto=True)
    ax.set_ylim(*check_bounds(y[j], ymin, ymax), auto=True)
    return contf, cont


def kde_contour_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot a 2d marginalised distribution as contours.

    This functions as a wrapper around matplotlib.axes.Axes.tricontour, and
    matplotlib.axes.Axes.tricontourf with a kernel density estimation
    computation provided by scipy.stats.gaussian_kde in between. All remaining
    keyword arguments are passed onwards to both functions.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axis object to plot on.

    data_x, data_y: np.array
        x and y coordinates of uniformly weighted samples to generate kernel
        density estimator.

    weights: np.array, optional
        Sample weights.

    ncompress: int, optional
        Degree of compression.
        optional, Default 1000

    xmin, xmax, ymin, ymax: float
        lower/upper prior bounds in x/y coordinates.
        optional, default None

    Returns
    -------
    c: matplotlib.contour.QuadContourSet
        A set of contourlines or filled regions

    """
    kwargs = normalize_kwargs(kwargs, dict(linewidths=['linewidth', 'lw'],
                                           linestyles=['linestyle', 'ls'],
                                           color=['c'],
                                           facecolor=['fc'],
                                           edgecolor=['ec']))
    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)
    ymin = kwargs.pop('ymin', None)
    ymax = kwargs.pop('ymax', None)
    weights = kwargs.pop('weights', None)
    ncompress = kwargs.pop('ncompress', 1000)
    label = kwargs.pop('label', None)
    zorder = kwargs.pop('zorder', 1)
    color = kwargs.pop('color', next(ax._get_lines.prop_cycler)['color'])
    facecolor = kwargs.pop('facecolor', color)
    edgecolor = kwargs.pop(
        'edgecolor', color if facecolor in [None, 'None', 'none'] else 'k')
    kwargs.pop('q', None)

    if len(data_x) == 0 or len(data_y) == 0:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0))

    if weights is not None:
        data_x = data_x[weights != 0]
        data_y = data_y[weights != 0]
        weights = weights[weights != 0]

    cov = np.cov(data_x, data_y, aweights=weights)
    tri, w = triangular_sample_compression_2d(data_x, data_y, cov,
                                              weights, ncompress)
    kde = gaussian_kde([tri.x, tri.y], weights=w)

    x, y = kde.resample(ncompress)
    x = np.concatenate([tri.x, x])
    y = np.concatenate([tri.y, y])
    w = np.concatenate([w, np.zeros(ncompress)])
    tri = scaled_triangulation(x, y, cov)

    p = kde([tri.x, tri.y])

    sigmax = np.sqrt(kde.covariance[0, 0])
    p = cut_and_normalise_gaussian(tri.x, p, sigmax, xmin, xmax)
    sigmay = np.sqrt(kde.covariance[1, 1])
    p = cut_and_normalise_gaussian(tri.y, p, sigmay, ymin, ymax)

    contours = iso_probability_contours_from_samples(p, weights=w)

    if facecolor not in [None, 'None', 'none']:
        linewidths = kwargs.pop('linewidths', 0.5)
        cmap = kwargs.pop('cmap', basic_cmap(facecolor))
        contf = ax.tricontourf(tri, p, contours, cmap=cmap, zorder=zorder,
                               vmin=0, vmax=p.max(), *args, **kwargs)
        for c in contf.collections:
            c.set_cmap(cmap)
        ax.patches += [plt.Rectangle((0, 0), 0, 0, lw=2, label=label,
                                     fc=cmap(0.999), ec=cmap(0.32))]
        cmap = None
    else:
        linewidths = kwargs.pop('linewidths',
                                plt.rcParams.get('lines.linewidth'))
        cmap = kwargs.pop('cmap', None)
        contf = None
        ax.patches += [
            plt.Rectangle((0, 0), 0, 0, lw=2, label=label,
                          fc='None' if cmap is None else cmap(0.999),
                          ec=edgecolor if cmap is None else cmap(0.32))
        ]
        edgecolor = edgecolor if cmap is None else None

    vmin, vmax = match_contour_to_contourf(contours, vmin=0, vmax=p.max())
    cont = ax.tricontour(tri, p, contours, zorder=zorder,
                         vmin=vmin, vmax=vmax, linewidths=linewidths,
                         colors=edgecolor, cmap=cmap, *args, **kwargs)

    ax.set_xlim(*check_bounds(tri.x, xmin, xmax), auto=True)
    ax.set_ylim(*check_bounds(tri.y, ymin, ymax), auto=True)
    return contf, cont


def hist_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot a 2d marginalised distribution as a histogram.

    This functions as a wrapper around matplotlib.axes.Axes.hist2d

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axis object to plot on

    data_x, data_y: np.array
        x and y coordinates of uniformly weighted samples to generate kernel
        density estimator.

    xmin, xmax, ymin, ymax: float
        lower/upper prior bounds in x/y coordinates
        optional, default None

    levels: list
        Shade iso-probability contours containing these levels of probability
        mass. If None defaults to usual matplotlib.axes.Axes.hist2d colouring.
        optional, default None

    Returns
    -------
    c: matplotlib.collections.QuadMesh
        A set of colors

    """
    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)
    ymin = kwargs.pop('ymin', None)
    ymax = kwargs.pop('ymax', None)
    label = kwargs.pop('label', None)
    levels = kwargs.pop('levels', None)
    color = kwargs.pop('color', next(ax._get_lines.prop_cycler)['color'])
    weights = kwargs.pop('weights', None)

    if xmin is None or not np.isfinite(xmin):
        xmin = quantile(data_x, 0.01, weights)
    if xmax is None or not np.isfinite(xmax):
        xmax = quantile(data_x, 0.99, weights)
    if ymin is None or not np.isfinite(ymin):
        ymin = quantile(data_y, 0.01, weights)
    if ymax is None or not np.isfinite(ymax):
        ymax = quantile(data_y, 0.99, weights)

    rge = kwargs.pop('range', ((xmin, xmax), (ymin, ymax)))

    if len(data_x) == 0 or len(data_y) == 0:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0))

    cmap = kwargs.pop('cmap', basic_cmap(color))

    if levels is None:
        pdf, x, y, image = ax.hist2d(data_x, data_y, weights=weights,
                                     cmap=cmap, range=rge,
                                     *args, **kwargs)
    else:
        bins = kwargs.pop('bins', 10)
        density = kwargs.pop('density', False)
        cmin = kwargs.pop('cmin', None)
        cmax = kwargs.pop('cmax', None)
        pdf, x, y = np.histogram2d(data_x, data_y, bins, rge,
                                   density, weights)
        levels = iso_probability_contours(pdf, levels)
        pdf = np.digitize(pdf, levels, right=True)
        pdf = np.array(levels)[pdf]
        pdf = np.ma.masked_array(pdf, pdf < levels[1])
        if cmin is not None:
            pdf[pdf < cmin] = np.ma.masked
        if cmax is not None:
            pdf[pdf > cmax] = np.ma.masked
        image = ax.pcolormesh(x, y, pdf.T, cmap=cmap, vmin=0, vmax=pdf.max(),
                              *args, **kwargs)

    ax.patches += [plt.Rectangle((0, 0), 0, 0, fc=cmap(0.999), ec=cmap(0.32),
                                 lw=2, label=label)]

    ax.set_xlim(*check_bounds(x, xmin, xmax), auto=True)
    ax.set_ylim(*check_bounds(y, ymin, ymax), auto=True)
    return image


def scatter_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot samples from a 2d marginalised distribution.

    This functions as a wrapper around matplotlib.axes.Axes.plot, enforcing any
    prior bounds. All remaining keyword arguments are passed onwards.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        axis object to plot on

    data_x, data_y: np.array
        x and y coordinates of uniformly weighted samples to plot.

    xmin, xmax, ymin, ymax: float
        lower/upper prior bounds in x/y coordinates
        optional, default None

    Returns
    -------
    lines: matplotlib.lines.Line2D
        A list of line objects representing the plotted data (same as
        matplotlib matplotlib.axes.Axes.plot command)

    """
    kwargs = normalize_kwargs(
        kwargs,
        dict(color=['c'], mfc=['facecolor', 'fc'], mec=['edgecolor', 'ec']),
        drop=['ls', 'lw'])
    kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)
    ymin = kwargs.pop('ymin', None)
    ymax = kwargs.pop('ymax', None)
    kwargs.pop('q', None)
    markersize = kwargs.pop('markersize', 1)
    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', (next(ax._get_lines.prop_cycler)['color']
                                 if cmap is None else cmap(0.68)))

    points = ax.plot(data_x, data_y, 'o', color=color, markersize=markersize,
                     *args, **kwargs)
    ax.set_xlim(*check_bounds(data_x, xmin, xmax), auto=True)
    ax.set_ylim(*check_bounds(data_y, ymin, ymax), auto=True)
    return points


def basic_cmap(color):
    """Construct basic colormap a single color."""
    return LinearSegmentedColormap.from_list(color, ['#ffffff', color])


def make_diagonal(ax):
    """Link x and y axes limits."""
    class DiagonalAxes(type(ax)):
        def set_xlim(self, left=None, right=None, emit=True, auto=False,
                     xmin=None, xmax=None):
            super().set_ylim(bottom=left, top=right, emit=True, auto=auto,
                             ymin=xmin, ymax=xmax)
            return super().set_xlim(left=left, right=right, emit=emit,
                                    auto=auto, xmin=xmin, xmax=xmax)

        def set_ylim(self, bottom=None, top=None, emit=True, auto=False,
                     ymin=None, ymax=None):
            super().set_xlim(left=bottom, right=top, emit=True, auto=auto,
                             xmin=ymin, xmax=ymax)
            return super().set_ylim(bottom=bottom, top=top, emit=emit,
                                    auto=auto, ymin=ymin, ymax=ymax)
    ax.__class__ = DiagonalAxes


def quantile_plot_interval(q):
    """Interpret quantile q input to quantile plot range tuple."""
    if isinstance(q, str):
        sigmas = {'1sigma': 0.682689492137086,
                  '2sigma': 0.954499736103642,
                  '3sigma': 0.997300203936740,
                  '4sigma': 0.999936657516334,
                  '5sigma': 0.999999426696856}
        q = (1 - sigmas[q]) / 2
    if isinstance(q, float) or isinstance(q, int):
        if q > 0.5:
            q = 1 - q
        q = (q, 1-q)
    return q


def normalize_kwargs(kwargs, alias_mapping=None, drop=None):
    """Normalize kwarg inputs.

    Works the same way as cbook.normalize_kwargs, but additionally allows to
    drop kwargs.
    """
    drop = [] if drop is None else drop
    alias_mapping = {} if alias_mapping is None else alias_mapping
    kwargs = cbook.normalize_kwargs(kwargs, alias_mapping=alias_mapping)
    for key in set(drop) & set(kwargs.keys()):
        kwargs.pop(key)
    return kwargs
