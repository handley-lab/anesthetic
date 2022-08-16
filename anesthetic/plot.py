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
from scipy.special import erf
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
from anesthetic.utils import nest_level
from anesthetic.utils import (sample_compression_1d, quantile,
                              triangular_sample_compression_2d,
                              iso_probability_contours,
                              scaled_triangulation, match_contour_to_contourf)
from anesthetic.boundary import cut_and_normalise_gaussian


class AxesSeries(pandas.Series):
    """Anesthetic's axes version of `pandas.Series`."""

    @property
    def _constructor(self):
        return AxesSeries

    @property
    def _constructor_expanddim(self):
        return AxesDataFrame


class AxesDataFrame(pandas.DataFrame):
    """Anesthetic's axes version of `pandas.DataFrame`."""

    @property
    def _constructor(self):
        return AxesDataFrame

    @property
    def _constructor_sliced(self):
        return AxesSeries

    def axlines(self, params, values, **kwargs):
        """Add vertical and horizontal lines across all axes.

        Parameters
        ----------
            params : str or list(str)
                parameter label(s).
                Should match the size of `values`.
            values : float or list(float)
                value(s) at which vertical and horizontal lines shall be added.
                Should match the size of `params`.
            kwargs
                Any kwarg that can be passed to `plt.axvline` or `plt.axhline`.

        """
        params = np.ravel(params)
        values = np.ravel(values)
        if params.size != values.size:
            raise ValueError("The sizes of `params` and `values` must match "
                             "exactly, but params.size=%s and values.size=%s."
                             % (params.size, values.size))
        for i, param in enumerate(params):
            if param in self.columns:
                for ax in self.loc[:, param]:
                    if ax is not None:
                        ax.axvline(values[i], **kwargs)
            if param in self.index:
                for ax in self.loc[param, self.columns != param]:
                    if ax is not None:
                        ax.axhline(values[i], **kwargs)

    def axspans(self, params, vmins, vmaxs, **kwargs):
        """Add vertical and horizontal spans across all axes.

        Parameters
        ----------
            params : str or list(str)
                parameter label(s).
                Should match the size of `vmins` and `vmaxs`.
            vmins : float or list(float)
                Minimum value of the vertical and horizontal axes spans.
                Should match the size of `params`.
            vmaxs : float or list(float)
                Maximum value of the vertical and horizontal axes spans.
                Should match the size of `params`.
            kwargs
                Any kwarg that can be passed to `plt.axvspan` or `plt.axhspan`.

        """
        kwargs = normalize_kwargs(kwargs, dict(color=['c']))
        params = np.ravel(params)
        vmins = np.ravel(vmins)
        vmaxs = np.ravel(vmaxs)
        if params.size != vmins.size:
            raise ValueError("The sizes of `params`, `vmins` and `vmaxs` must "
                             "match exactly, but params.size=%s, "
                             "vmins.size=%s and vmaxs.size=%s."
                             % (params.size, vmins.size, vmaxs.size))
        for i, param in enumerate(params):
            if param in self.columns:
                for ax in self.loc[:, param]:
                    if ax is not None:
                        ax.axvspan(vmins[i], vmaxs[i], **kwargs)
            if param in self.index:
                for ax in self.loc[param, self.columns != param]:
                    if ax is not None:
                        ax.axhspan(vmins[i], vmaxs[i], **kwargs)


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
    axes = AxesSeries(index=np.atleast_1d(params), dtype=object)
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

        ticks: str
            If 'outer', plot ticks only on the very left and very bottom.
            If 'inner', plot ticks also in inner subplots.
            If None, plot no ticks at all.
            Default: 'outer'

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

    ticks = kwargs.pop('ticks', 'outer')
    upper = kwargs.pop('upper', True)
    lower = kwargs.pop('lower', True)
    diagonal = kwargs.pop('diagonal', True)

    axes = AxesDataFrame(index=np.atleast_1d(yparams),
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
    for j, y in enumerate(axes.index[::-1]):
        for i, x in enumerate(axes.columns):
            if position[x][y] is not None:
                sx = list(axes[x].dropna())
                sx = sx[0] if sx else None
                sy = list(axes.T[y].dropna())
                sy = sy[0] if sy else None
                axes[x][y] = fig.add_subplot(grid[axes.index.size-1-j, i],
                                             sharex=sx, sharey=sy)

                if position[x][y] == 0:
                    axes[x][y].twin = axes[x][y].twinx()
                    axes[x][y].twin.set_yticks([])
                    axes[x][y].twin.set_ylim(0, 1.1)
                    make_diagonal(axes[x][y])
                    axes[x][y].position = 'diagonal'
                    axes[x][y].twin.xaxis.set_major_locator(
                        MaxNLocator(3, prune='both'))
                else:
                    if position[x][y] == 1:
                        axes[x][y].position = 'upper'
                    elif position[x][y] == -1:
                        axes[x][y].position = 'lower'
                    axes[x][y].yaxis.set_major_locator(
                        MaxNLocator(3, prune='both'))
                axes[x][y].xaxis.set_major_locator(
                    MaxNLocator(3, prune='both'))

    for y, ax in axes.bfill(axis=1).iloc[:, 0].dropna().iteritems():
        ax.set_ylabel(tex[y])

    for x, ax in axes.ffill(axis=0).iloc[-1, :].dropna().iteritems():
        ax.set_xlabel(tex[x])

    # left and right ticks and labels
    for y, ax in axes.iterrows():
        ax_ = ax.dropna()
        if len(ax_) and ticks == 'inner':
            for i, a in enumerate(ax_):
                if i == 0:  # first column
                    if a.position == 'diagonal' and len(ax_) == 1:
                        a.tick_params('y', left=False, labelleft=False)
                    else:
                        a.tick_params('y', left=True, labelleft=True)
                elif a.position == 'diagonal':  # not first column
                    tl = a.yaxis.majorTicks[0].tick1line.get_markersize()
                    a.tick_params('y', direction='out', length=tl/2,
                                  left=True, labelleft=False)
                else:  # not diagonal and not first column
                    a.tick_params('y', direction='inout',
                                  left=True, labelleft=False)
        elif len(ax_) and ticks == 'outer':  # no inner ticks
            for a in ax_[1:]:
                a.tick_params('y', left=False, labelleft=False)
        elif len(ax_) and ticks is None:  # no ticks at all
            for a in ax_:
                a.tick_params('y', left=False, right=False,
                              labelleft=False, labelright=False)
        else:
            raise ValueError(
                "ticks=%s was requested, but ticks can only be one of "
                "['outer', 'inner', None]." % ticks)

    # bottom and top ticks and labels
    for x, ax in axes.iteritems():
        ax_ = ax.dropna()
        if len(ax_):
            if ticks == 'inner':
                for i, a in enumerate(ax_):
                    if i == len(ax_) - 1:  # bottom row
                        a.tick_params('x', bottom=True, labelbottom=True)
                    else:  # not bottom row
                        a.tick_params('x', direction='inout',
                                      bottom=True, labelbottom=False)
                        if a.position == 'diagonal':
                            a.twin.tick_params('x', direction='inout',
                                               bottom=True, labelbottom=False)
            elif ticks == 'outer':  # no inner ticks
                for a in ax_[:-1]:
                    a.tick_params('x', bottom=False, labelbottom=False)
            elif ticks is None:  # no ticks at all
                for a in ax_:
                    a.tick_params('x', bottom=False, top=False,
                                  labelbottom=False, labeltop=False)

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

    levels: list
        values at which to draw iso-probability lines.
        optional, default [0.95, 0.68]

    q: int or float or tuple
        Quantile to determine the data range to be plotted.
        - 0: full data range, i.e. q=0 --> quantile range (0, 1)
        - int: `q`-sigma data range, e.g. q=1 --> quantile range (0.16, 0.84)
        - float: percentile, e.g. q=0.68 --> quantile range  (0.16, 0.84)
        - tuple: quantile range, e.g. (0.16, 0.84)
        Default 5

    facecolor: bool or string
        If set to True then the 1d plot will be shaded with the value of the
        ``color`` kwarg. Set to a string such as 'blue', 'k', 'r', 'C1' ect.
        to define the color of the shading directly.
        optional, default False

    Returns
    -------
    lines: matplotlib.lines.Line2D
        A list of line objects representing the plotted data (same as
        matplotlib matplotlib.axes.Axes.plot command)

    """
    kwargs = normalize_kwargs(
        kwargs,
        dict(linewidth=['lw'], linestyle=['ls'], color=['c'],
             facecolor=['fc'], edgecolor=['ec']))

    if data.max()-data.min() <= 0:
        return

    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)
    levels = kwargs.pop('levels', [0.95, 0.68])
    density = kwargs.pop('density', False)

    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', (next(ax._get_lines.prop_cycler)['color']
                                 if cmap is None else cmap(0.68)))
    facecolor = kwargs.pop('facecolor', False)
    if 'edgecolor' in kwargs:
        edgecolor = kwargs.pop('edgecolor')
        if edgecolor:
            color = edgecolor
    else:
        edgecolor = color

    q = kwargs.pop('q', 5)
    q = quantile_plot_interval(q=q)

    try:
        x, p, xmin, xmax = fastkde_1d(data, xmin, xmax)
    except NameError:
        raise ImportError("You need to install fastkde to use fastkde")
    p /= p.max()
    i = ((x > quantile(x, q[0], p)) & (x < quantile(x, q[-1], p)))

    area = np.trapz(x=x[i], y=p[i]) if density else 1
    ans = ax.plot(x[i], p[i]/area, color=color, *args, **kwargs)

    if facecolor and facecolor not in [None, 'None', 'none']:
        if facecolor is True:
            facecolor = color
        c = iso_probability_contours(p[i], contours=levels)
        cmap = basic_cmap(facecolor)
        fill = []
        for j in range(len(c)-1):
            fill.append(ax.fill_between(x[i], p[i], where=p[i] >= c[j],
                        color=cmap(c[j]), edgecolor=edgecolor))

        return ans, fill

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
        Degree of compression.
        Default 10000

    nplot: int, optional
        Number of plotting points to use.
        Default 100

    levels: list
        values at which to draw iso-probability lines.
        optional, default [0.95, 0.68]

    q: int or float or tuple
        Quantile to determine the data range to be plotted.
        - 0: full data range, i.e. q=0 --> quantile range (0, 1)
        - int: `q`-sigma data range, e.g. q=1 --> quantile range (0.16, 0.84)
        - float: percentile, e.g. q=0.68 --> quantile range  (0.16, 0.84)
        - tuple: quantile range, e.g. (0.16, 0.84)
        Default 5

    facecolor: bool or string
        If set to True then the 1d plot will be shaded with the value of the
        ``color`` kwarg. Set to a string such as 'blue', 'k', 'r', 'C1' ect.
        to define the color of the shading directly.
        optional, default False

    Returns
    -------
    lines: matplotlib.lines.Line2D
        A list of line objects representing the plotted data (same as
        matplotlib matplotlib.axes.Axes.plot command)

    """
    kwargs = normalize_kwargs(
        kwargs,
        dict(linewidth=['lw'], linestyle=['ls'], color=['c'],
             facecolor=['fc'], edgecolor=['ec']))

    if data.max()-data.min() <= 0:
        return

    weights = kwargs.pop('weights', None)
    if weights is not None:
        data = data[weights != 0]
        weights = weights[weights != 0]

    ncompress = kwargs.pop('ncompress', 10000)
    nplot = kwargs.pop('nplot', 100)
    bw_method = kwargs.pop('bw_method', None)
    levels = kwargs.pop('levels', [0.95, 0.68])
    density = kwargs.pop('density', False)

    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', (next(ax._get_lines.prop_cycler)['color']
                                 if cmap is None else cmap(0.68)))
    facecolor = kwargs.pop('facecolor', False)
    if 'edgecolor' in kwargs:
        edgecolor = kwargs.pop('edgecolor')
        if edgecolor:
            color = edgecolor
    else:
        edgecolor = color

    q = kwargs.pop('q', 5)
    q = quantile_plot_interval(q=q)

    data_compressed, w = sample_compression_1d(data, weights, ncompress)
    kde = gaussian_kde(data_compressed, weights=w, bw_method=bw_method)
    xmin = quantile(data, q[0], weights)
    xmax = quantile(data, q[-1], weights)
    x = np.linspace(xmin, xmax, nplot)

    p = kde(x)
    p /= p.max()
    bw = np.sqrt(kde.covariance[0, 0])
    pp = cut_and_normalise_gaussian(x, p, bw, xmin=data.min(), xmax=data.max())
    pp /= pp.max()
    area = np.trapz(x=x, y=pp) if density else 1
    ans = ax.plot(x, pp/area, color=color, *args, **kwargs)

    if facecolor and facecolor not in [None, 'None', 'none']:
        if facecolor is True:
            facecolor = color
        c = iso_probability_contours(pp, contours=levels)
        cmap = basic_cmap(facecolor)
        fill = []
        for j in range(len(c)-1):
            fill.append(ax.fill_between(x, pp, where=pp >= c[j],
                        color=cmap(c[j]), edgecolor=edgecolor))

        return ans, fill

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

    q: int or float or tuple
        Quantile to determine the data range to be plotted.
        - 0: full data range, i.e. q=0 --> quantile range (0, 1)
        - int: `q`-sigma data range, e.g. q=1 --> quantile range (0.16, 0.84)
        - float: percentile, e.g. q=0.68 --> quantile range  (0.16, 0.84)
        - tuple: quantile range, e.g. (0.16, 0.84)
        Default 5

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

    weights = kwargs.pop('weights', None)
    bins = kwargs.pop('bins', 10)
    histtype = kwargs.pop('histtype', 'bar')
    density = kwargs.get('density', False)

    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', (next(ax._get_lines.prop_cycler)['color']
                                 if cmap is None else cmap(0.68)))

    q = kwargs.pop('q', 5)
    q = quantile_plot_interval(q=q)
    xmin = quantile(data, q[0], weights)
    xmax = quantile(data, q[-1], weights)

    if bins in ['knuth', 'freedman', 'blocks']:
        try:
            h, edges, bars = hist(data, ax=ax, bins=bins,
                                  range=(xmin, xmax), histtype=histtype,
                                  color=color, *args, **kwargs)
        except NameError:
            raise ImportError("You need to install astropy to use astropyhist")
    else:
        h, edges, bars = ax.hist(data, weights=weights, bins=bins,
                                 range=(xmin, xmax), histtype=histtype,
                                 color=color, *args, **kwargs)

    if histtype == 'bar' and not density:
        for b in bars:
            b.set_height(b.get_height() / h.max())
    elif (histtype == 'step' or histtype == 'stepfilled') and not density:
        trans = Affine2D().scale(sx=1, sy=1./h.max()) + ax.transData
        bars[0].set_transform(trans)

    if not density:
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
        Has to be ordered from outermost to innermost contour.
        optional, default [0.95, 0.68]

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
    levels = kwargs.pop('levels', [0.95, 0.68])

    color = kwargs.pop('color', next(ax._get_lines.prop_cycler)['color'])
    facecolor = kwargs.pop('facecolor', True)
    edgecolor = kwargs.pop('edgecolor', None)
    cmap = kwargs.pop('cmap', None)
    facecolor, edgecolor, cmap = set_colors(c=color, fc=facecolor,
                                            ec=edgecolor, cmap=cmap)

    kwargs.pop('q', None)

    try:
        x, y, pdf, xmin, xmax, ymin, ymax = fastkde_2d(data_x, data_y,
                                                       xmin=xmin, xmax=xmax,
                                                       ymin=ymin, ymax=ymax)
    except NameError:
        raise ImportError("You need to install fastkde to use fastkde")

    levels = iso_probability_contours(pdf, contours=levels)

    i = (pdf >= levels[0]*0.5).any(axis=0)
    j = (pdf >= levels[0]*0.5).any(axis=1)

    if facecolor not in [None, 'None', 'none']:
        linewidths = kwargs.pop('linewidths', 0.5)
        contf = ax.contourf(x[i], y[j], pdf[np.ix_(j, i)], levels, cmap=cmap,
                            zorder=zorder, vmin=0, vmax=pdf.max(),
                            *args, **kwargs)
        for c in contf.collections:
            c.set_cmap(cmap)
        ax.add_patch(plt.Rectangle((0, 0), 0, 0, lw=2, label=label,
                                   fc=cmap(0.999), ec=cmap(0.32)))
        cmap = None
    else:
        linewidths = kwargs.pop('linewidths',
                                plt.rcParams.get('lines.linewidth'))
        contf = None
        ax.add_patch(
            plt.Rectangle((0, 0), 0, 0, lw=2, label=label,
                          fc='None' if cmap is None else cmap(0.999),
                          ec=edgecolor if cmap is None else cmap(0.32))
        )

    vmin, vmax = match_contour_to_contourf(levels, vmin=0, vmax=pdf.max())
    cont = ax.contour(x[i], y[j], pdf[np.ix_(j, i)], levels, zorder=zorder,
                      vmin=vmin, vmax=vmax, linewidths=linewidths,
                      colors=edgecolor, cmap=cmap, *args, **kwargs)

    ax.set_xlim(xmin, xmax, auto=True)
    ax.set_ylim(ymin, ymax, auto=True)
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

    levels: list, optional
        amount of mass within each iso-probability contour.
        Has to be ordered from outermost to innermost contour.
        optional, default [0.95, 0.68]

    ncompress: int, optional
        Degree of compression.
        Default 10000

    nplot: int, optional
        Number of plotting points to use.
        Default 1000

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

    weights = kwargs.pop('weights', None)
    if weights is not None:
        data_x = data_x[weights != 0]
        data_y = data_y[weights != 0]
        weights = weights[weights != 0]

    ncompress = kwargs.pop('ncompress', 10000)
    nplot = kwargs.pop('nplot', 1000)
    bw_method = kwargs.pop('bw_method', None)
    label = kwargs.pop('label', None)
    zorder = kwargs.pop('zorder', 1)
    levels = kwargs.pop('levels', [0.95, 0.68])

    color = kwargs.pop('color', next(ax._get_lines.prop_cycler)['color'])
    facecolor = kwargs.pop('facecolor', True)
    edgecolor = kwargs.pop('edgecolor', None)
    cmap = kwargs.pop('cmap', None)
    facecolor, edgecolor, cmap = set_colors(c=color, fc=facecolor,
                                            ec=edgecolor, cmap=cmap)

    kwargs.pop('q', None)

    cov = np.cov(data_x, data_y, aweights=weights)
    tri, w = triangular_sample_compression_2d(data_x, data_y, cov,
                                              weights, ncompress)
    kde = gaussian_kde([tri.x, tri.y], weights=w, bw_method=bw_method)

    q = kwargs.pop('q', 5)
    q = quantile_plot_interval(q=q)
    xmin = quantile(data_x, q[0], weights)
    xmax = quantile(data_x, q[-1], weights)
    ymin = quantile(data_y, q[0], weights)
    ymax = quantile(data_y, q[-1], weights)
    X, Y = np.mgrid[xmin:xmax:1j*np.sqrt(nplot), ymin:ymax:1j*np.sqrt(nplot)]

    P = kde([X.ravel(), Y.ravel()]).reshape(X.shape)

    bw_x = np.sqrt(kde.covariance[0, 0])
    P = cut_and_normalise_gaussian(X, P, bw=bw_x,
                                   xmin=data_x.min(), xmax=data_x.max())
    bw_y = np.sqrt(kde.covariance[1, 1])
    P = cut_and_normalise_gaussian(Y, P, bw=bw_y,
                                   xmin=data_y.min(), xmax=data_y.max())

    levels = iso_probability_contours(P, contours=levels)

    if facecolor not in [None, 'None', 'none']:
        linewidths = kwargs.pop('linewidths', 0.5)
        contf = ax.contourf(X, Y, P, levels=levels, cmap=cmap, zorder=zorder,
                            vmin=0, vmax=P.max(), *args, **kwargs)
        for c in contf.collections:
            c.set_cmap(cmap)
        ax.add_patch(plt.Rectangle((0, 0), 0, 0, lw=2, label=label,
                                   fc=cmap(0.999), ec=cmap(0.32)))
        cmap = None
    else:
        linewidths = kwargs.pop('linewidths',
                                plt.rcParams.get('lines.linewidth'))
        contf = None
        ax.add_patch(
            plt.Rectangle((0, 0), 0, 0, lw=2, label=label,
                          fc='None' if cmap is None else cmap(0.999),
                          ec=edgecolor if cmap is None else cmap(0.32))
        )

    vmin, vmax = match_contour_to_contourf(levels, vmin=0, vmax=P.max())
    cont = ax.contour(X, Y, P, levels=levels, zorder=zorder,
                      vmin=vmin, vmax=vmax, linewidths=linewidths,
                      colors=edgecolor, cmap=cmap, *args, **kwargs)

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

    levels: list
        Shade iso-probability contours containing these levels of probability
        mass. If None defaults to usual matplotlib.axes.Axes.hist2d colouring.
        optional, default None

    q: int or float or tuple
        Quantile to determine the data range to be plotted.
        - 0: full data range, i.e. q=0 --> quantile range (0, 1)
        - int: `q`-sigma data range, e.g. q=1 --> quantile range (0.16, 0.84)
        - float: percentile, e.g. q=0.68 --> quantile range  (0.16, 0.84)
        - tuple: quantile range, e.g. (0.16, 0.84)
        Default 5

    Returns
    -------
    c: matplotlib.collections.QuadMesh
        A set of colors

    """
    weights = kwargs.pop('weights', None)

    vmin = kwargs.pop('vmin', 0)
    label = kwargs.pop('label', None)
    levels = kwargs.pop('levels', None)

    color = kwargs.pop('color', next(ax._get_lines.prop_cycler)['color'])
    cmap = kwargs.pop('cmap', basic_cmap(color))

    q = kwargs.pop('q', 5)
    q = quantile_plot_interval(q=q)
    xmin = quantile(data_x, q[0], weights)
    xmax = quantile(data_x, q[-1], weights)
    ymin = quantile(data_y, q[0], weights)
    ymax = quantile(data_y, q[-1], weights)
    rge = kwargs.pop('range', ((xmin, xmax), (ymin, ymax)))

    if levels is None:
        pdf, x, y, image = ax.hist2d(data_x, data_y, weights=weights,
                                     cmap=cmap, range=rge, vmin=vmin,
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
        image = ax.pcolormesh(x, y, pdf.T, cmap=cmap, vmin=vmin,
                              *args, **kwargs)

    ax.add_patch(plt.Rectangle((0, 0), 0, 0, fc=cmap(0.999), ec=cmap(0.32),
                               lw=2, label=label))

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

    Returns
    -------
    lines: matplotlib.lines.Line2D
        A list of line objects representing the plotted data (same as
        matplotlib.axes.Axes.plot command)

    """
    kwargs = normalize_kwargs(
        kwargs,
        dict(color=['c'], mfc=['facecolor', 'fc'], mec=['edgecolor', 'ec']),
        drop=['ls', 'lw'])
    kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)

    markersize = kwargs.pop('markersize', 1)
    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', (next(ax._get_lines.prop_cycler)['color']
                                 if cmap is None else cmap(0.68)))

    kwargs.pop('q', None)

    points = ax.plot(data_x, data_y, 'o', color=color, markersize=markersize,
                     *args, **kwargs)
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

        def get_legend_handles_labels(self, *args, **kwargs):
            return self.twin.get_legend_handles_labels(*args, **kwargs)

        def legend(self, *args, **kwargs):
            return self.twin.legend(*args, **kwargs)

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
    elif isinstance(q, int) and q >= 1:
        q = (1 - erf(q / np.sqrt(2))) / 2
    if isinstance(q, float) or isinstance(q, int):
        if q > 0.5:
            q = 1 - q
        q = (q, 1-q)
    return tuple(np.sort(q))


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


def set_colors(c, fc, ec, cmap):
    """Navigate interplay between possible color inputs {c, fc, ec, cmap}."""
    if fc in [None, 'None', 'none']:
        # unfilled contours
        if ec is None and cmap is None:
            cmap = basic_cmap(c)
    else:
        # filled contours
        if fc is True:
            fc = c
        if ec is None and cmap is None:
            ec = c
            cmap = basic_cmap(fc)
        elif ec is None:
            ec = (cmap(1.),)
        elif cmap is None:
            cmap = basic_cmap(fc)
    return fc, ec, cmap
