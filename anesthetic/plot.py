"""Lower-level plotting tools.

Routines that may be of use to users wishing for more fine-grained control may
wish to use.

- :func:`anesthetic.plot.make_1d_axes`
- :func:`anesthetic.plot.make_2d_axes`

to create a set of axes and legend proxies.

"""
from packaging import version
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.special import erf
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.axes import Axes
import matplotlib.cbook as cbook
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, LogLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Affine2D
from anesthetic.utils import nest_level
from anesthetic.utils import (sample_compression_1d, quantile,
                              triangular_sample_compression_2d,
                              iso_probability_contours,
                              iso_probability_contours_from_samples,
                              match_contour_to_contourf, histogram_bin_edges)
from anesthetic.boundary import boundary_correction_1d, boundary_correction_2d


class AxesSeries(Series):
    """Anesthetic's axes version of :class:`pandas.Series`.

    Parameters
    ----------
    index : list(str)
        Parameters to be placed on the y-axes.
    fig : :class:`matplotlib.figure.Figure`
    ncol : int
        Number of axes columns. Decides after how many axes the AxesSeries is
        split to continue in a new row.
    labels : dict(str:str), optional
        Dictionary mapping params to plot labels.
        Default: params
    logx : list(str), optional
        List of parameters to be plotted on a log scale.
    gridspec_kw : dict, optional
        Dict with keywords passed to the :class:`matplotlib.gridspec.GridSpec`
        constructor used to create the grid the subplots are placed on.
    subplot_spec : :class:`matplotlib.gridspec.GridSpec`, default=None
        GridSpec instance to plot array as part of a subfigure.

    Methods
    -------
    set_xlabels:
        Set the labels for the x-axes.
    tick_params:
        Set tick parameters across all axes.

    """

    _metadata = ['_logx']
    _logx = []

    def __init__(self, data=None, index=None, fig=None, ncol=None, labels=None,
                 logx=None, gridspec_kw=None, subplot_spec=None,
                 *args, **kwargs):
        if data is None and index is not None:
            data = self.axes_series(index=index, fig=fig, ncol=ncol,
                                    gridspec_kw=gridspec_kw,
                                    subplot_spec=subplot_spec)
            self._set_xlabels(axes=data, labels=labels)
        super().__init__(data=data, index=index, *args, **kwargs)
        if logx is None:
            self._logx = []
        else:
            self._logx = logx
            self._set_xscale()

    @property
    def _constructor(self):
        return AxesSeries

    @property
    def _constructor_expanddim(self):
        return AxesDataFrame

    @staticmethod
    def axes_series(index, fig, ncol=None, gridspec_kw=None,
                    subplot_spec=None):
        """Set up subplots for :class:`AxesSeries`."""
        axes = Series(np.full(np.shape(index), None), index=index)
        if fig is None:
            fig = plt.figure()
        if ncol is None:
            ncol = int(np.ceil(np.sqrt(axes.index.size)))
        ncol = ncol
        nrow = int(np.ceil(axes.index.size / ncol))
        if gridspec_kw is None:
            gridspec_kw = {}
        wspace = gridspec_kw.pop('wspace', 0)
        if subplot_spec is None:
            gs = GridSpec(nrow, ncol, wspace=wspace, **gridspec_kw)
        else:
            gs = GridSpecFromSubplotSpec(nrow, ncol, wspace=wspace,
                                         subplot_spec=subplot_spec,
                                         **gridspec_kw)
        for p, g in zip(axes.index, gs):
            axes[p] = ax = fig.add_subplot(g)
            ax.set_yticks([])
        return axes

    def _set_xscale(self):
        for p, ax in self.items():
            if p in self._logx:
                ax.set_xscale('log')

    @staticmethod
    def _set_xlabels(axes, labels, **kwargs):
        if labels is None:
            labels = {}
        labels = {p: labels[p] if p in labels else p for p in axes.index}
        for p, ax in axes.items():
            ax.set_xlabel(labels[p], **kwargs)

    def set_xlabels(self, labels, **kwargs):
        """Set the labels for the x-axes.

        Parameters
        ----------
            labels : dict
                Dictionary of the axes labels.
            kwargs
                Any kwarg that can be passed to
                :meth:`matplotlib.axes.Axes.set_xlabel`.

        """
        self._set_xlabels(axes=self, labels=labels, **kwargs)

    def tick_params(self, *args, **kwargs):
        """Apply :meth:`matplotlib.axes.Axes.tick_params` across all axes."""
        for p, ax in self.items():
            ax.tick_params(*args, **kwargs)


class AxesDataFrame(DataFrame):
    """Anesthetic's axes version of :class:`pandas.DataFrame`.

    Parameters
    ----------
    index : list(str)
        Parameters to be placed on the y-axes.
    columns : list(str)
        Parameters to be placed on the x-axes.
    fig : :class:`matplotlib.figure.Figure`
    lower, diagonal, upper : bool, default=True
        Whether to create 2D marginalised plots above or below the
        diagonal, or to create a 1D marginalised plot on the diagonal.
    labels : dict(str:str), optional
        Dictionary mapping params to plot labels.
        Default: params
    ticks : str, default='inner'
        If 'outer', plot ticks only on the very left and very bottom.
        If 'inner', plot ticks also in inner subplots.
        If None, plot no ticks at all.
    logx, logy : list(str), optional
        Lists of parameters to be plotted on a log scale on the x-axis or
        y-axis, respectively.
    gridspec_kw : dict, optional
        Dict with keywords passed to the :class:`matplotlib.gridspec.GridSpec`
        constructor used to create the grid the subplots are placed on.
    subplot_spec : :class:`matplotlib.gridspec.GridSpec`, default=None
        GridSpec instance to plot array as part of a subfigure.

    Methods
    -------
    axlines:
        Add vertical and horizontal lines across all axes.
    axspans:
        Add vertical and horizontal spans across all axes.
    scatter:
        Add scatter points across all axes.
    set_labels:
        Set the labels for the axes.
    set_margins:
        Set margins across all axes.
    tick_params:
        Set tick parameters across all axes.

    """

    _metadata = ['_logx', '_logy']
    _logx = []
    _logy = []

    def __init__(self, data=None, index=None, columns=None, fig=None,
                 lower=True, diagonal=True, upper=True, labels=None,
                 ticks='inner', logx=None, logy=None,
                 gridspec_kw=None, subplot_spec=None, *args, **kwargs):
        if data is None and index is not None and columns is not None:
            position = self._position_frame(index=index,
                                            columns=columns,
                                            lower=lower,
                                            diagonal=diagonal,
                                            upper=upper)
            data = self._axes_frame(position=position,
                                    fig=fig,
                                    gridspec_kw=gridspec_kw,
                                    subplot_spec=subplot_spec)
            self._set_labels(axes=data, labels=labels)
            index = data.index
            columns = data.columns
            self._tick_params(axes=data, direction=ticks, which='both')
        super().__init__(data=data,
                         index=index,
                         columns=columns,
                         *args, **kwargs)
        if logx is None:
            self._logx = []
        else:
            self._logx = logx
        if logy is None:
            self._logy = []
        else:
            self._logy = logy
        if self._logx or self._logy:
            self._set_scale()
        self.tick_params(axis='both', which='both', labelrotation=45,
                         labelsize='small')

    @property
    def _constructor(self):
        return AxesDataFrame

    @property
    def _constructor_sliced(self):
        return AxesSeries

    @staticmethod
    def _position_frame(index, columns, lower, diagonal, upper):
        """Compute positions with lower=-1, diagonal=0, upper=+1."""
        data = np.full((np.size(index), np.size(columns)), None)
        position = DataFrame(data=data, index=index, columns=columns)
        all_params = list(columns) + list(index)
        for j, y in enumerate(index):
            for i, x in enumerate(columns):
                if all_params.index(x) < all_params.index(y):
                    if lower:
                        position.loc[y, x] = -1
                elif all_params.index(x) > all_params.index(y):
                    if upper:
                        position.loc[y, x] = +1
                elif diagonal:
                    position.loc[y, x] = 0
        return position

    @classmethod
    def _axes_frame(cls, position, fig, gridspec_kw=None, subplot_spec=None):
        """Set up subplots for :class:`AxesDataFrame`."""
        axes = position.copy()
        axes.dropna(axis=0, how='all', inplace=True)
        axes.dropna(axis=1, how='all', inplace=True)
        if axes.size == 0:
            return axes
        if fig is None:
            fig = plt.figure()
        if gridspec_kw is None:
            gridspec_kw = {}
        hspace = gridspec_kw.pop('hspace', 0)
        wspace = gridspec_kw.pop('wspace', 0)
        if subplot_spec is None:
            gs = GridSpec(*axes.shape, hspace=hspace, wspace=wspace,
                          **gridspec_kw)
        else:
            gs = GridSpecFromSubplotSpec(*axes.shape,
                                         hspace=hspace, wspace=wspace,
                                         subplot_spec=subplot_spec,
                                         **gridspec_kw)
        axes.loc[:, :] = None
        for j, y in enumerate(axes.index[::-1]):
            for i, x in enumerate(axes.columns):
                if position[x][y] is not None:
                    sx = list(axes[x].dropna())
                    sx = sx[0] if sx else None
                    sy = list(axes.T[y].dropna())
                    sy = sy[0] if sy else None
                    axes.loc[y, x] = fig.add_subplot(
                        gs[axes.index.size - 1 - j, i], sharex=sx, sharey=sy
                    )
                    if position[x][y] == 0:
                        axes[x][y].twin = axes[x][y].twinx()
                        axes[x][y].twin.set_yticks([])
                        cls._make_diagonal(axes[x][y])
                        axes[x][y].position = 'diagonal'
                        axes[x][y].twin.xaxis.set_major_locator(
                            MaxNLocator(3, prune='both'))
                        axes[x][y].twin.xaxis.set_minor_locator(
                            AutoMinorLocator(1))
                        axes[x][y].yaxis.set_major_locator(
                            MaxNLocator(3, prune='both'))
                        axes[x][y].yaxis.set_minor_locator(AutoMinorLocator(1))
                    else:
                        if position[x][y] == 1:
                            axes[x][y].position = 'upper'
                            cls._make_offdiagonal(axes[x][y])
                        elif position[x][y] == -1:
                            axes[x][y].position = 'lower'
                            cls._make_offdiagonal(axes[x][y])
                        axes[x][y].yaxis.set_major_locator(
                            MaxNLocator(3, prune='both'))
                        axes[x][y].yaxis.set_minor_locator(AutoMinorLocator(1))
                    axes[x][y].xaxis.set_major_locator(
                        MaxNLocator(3, prune='both'))
                    axes[x][y].xaxis.set_minor_locator(AutoMinorLocator(1))
        return axes

    @staticmethod
    def _make_diagonal(ax):
        """Link x and y axes limits."""

        class DiagonalAxes(type(ax)):
            def set_xlim(self, left=None, right=None, emit=True, auto=False,
                         xmin=None, xmax=None):
                if (self.get_xaxis().get_scale() ==
                        self.get_yaxis().get_scale()):
                    super().set_ylim(bottom=left, top=right, emit=True,
                                     auto=auto, ymin=xmin, ymax=xmax)
                return super().set_xlim(left=left, right=right, emit=emit,
                                        auto=auto, xmin=xmin, xmax=xmax)

            def set_ylim(self, bottom=None, top=None, emit=True, auto=False,
                         ymin=None, ymax=None):
                if (self.get_xaxis().get_scale() ==
                        self.get_yaxis().get_scale()):
                    super().set_xlim(left=bottom, right=top, emit=True,
                                     auto=auto, xmin=ymin, xmax=ymax)
                return super().set_ylim(bottom=bottom, top=top, emit=emit,
                                        auto=auto, ymin=ymin, ymax=ymax)

            def get_legend_handles_labels(self, *args, **kwargs):
                return self.twin.get_legend_handles_labels(*args, **kwargs)

            def legend(self, *args, **kwargs):
                return self.twin.legend(*args, **kwargs)

        ax.__class__ = DiagonalAxes

    @staticmethod
    def _make_offdiagonal(ax):
        """Linking x to y axes limits in triangle plots."""

        class OffDiagonalAxes(type(ax)):
            def set_xlim(self, left=None, right=None, emit=True, auto=False,
                         xmin=None, xmax=None):
                left, right = super().set_xlim(left=left, right=right,
                                               emit=emit,
                                               auto=auto, xmin=xmin, xmax=xmax)
                if emit:
                    self.callbacks.process('xlim_changed', self)
                    # Call all other x-axes that are shared with this one
                    for other in self._shared_axes['x'].get_siblings(self):
                        if other is not self:
                            other.set_xlim(left, right, emit=False, auto=auto)
                return left, right

            def set_ylim(self, bottom=None, top=None, emit=True, auto=False,
                         ymin=None, ymax=None):
                bottom, top = super().set_ylim(bottom=bottom, top=top,
                                               emit=emit,
                                               auto=auto, ymin=ymin, ymax=ymax)
                if emit:
                    self.callbacks.process('ylim_changed', self)
                    # Call all other y-axes that are shared with this one
                    for other in self._shared_axes['y'].get_siblings(self):
                        if other is not self:
                            other.set_ylim(bottom, top, emit=False, auto=auto)
                return bottom, top

        ax.__class__ = OffDiagonalAxes

    def _set_scale(self):
        for y, rows in self.iterrows():
            for x, ax in rows.items():
                if ax is not None:
                    if x in self._logx:
                        ax.set_xscale('log')
                    if y in self._logy:
                        ax.set_yscale('log')

    def _set_logticks(self):
        for y, rows in self.iterrows():
            for x, ax in rows.items():
                if ax is not None:
                    if x in self._logx:
                        ax.xaxis.set_major_locator(LogLocator(numticks=3))
                        if x != y:
                            ax.set_xlim(ax.dataLim.intervalx)
                    if y in self._logy:
                        ax.yaxis.set_major_locator(LogLocator(numticks=3))
                        if y != x:
                            ax.set_ylim(ax.dataLim.intervaly)

    @staticmethod
    def _set_labels(axes, labels, **kwargs):
        all_params = list(axes.columns) + list(axes.index)
        if labels is None:
            labels = {}
        labels = {p: labels[p] if p in labels else p for p in all_params}

        for y, axes_row in axes.iterrows():
            if axes_row.size:
                axes_row.dropna(inplace=True)
                axes_row.iloc[0].set_ylabel(labels[y], **kwargs)

        for x, axes_col in axes.items():
            if axes_col.size:
                axes_col.dropna(inplace=True)
                axes_col.iloc[-1].set_xlabel(labels[x], **kwargs)

    def set_labels(self, labels, **kwargs):
        """Set the labels for the axes.

        Parameters
        ----------
        labels : dict
            Dictionary of the axes labels.
        kwargs
            Any kwarg that can be passed to
            :meth:`matplotlib.axes.Axes.set_xlabel` or
            :meth:`matplotlib.axes.Axes.set_ylabel`.

        """
        self._set_labels(axes=self, labels=labels, **kwargs)

    @staticmethod
    def _tick_params(axes, direction='inner', **kwargs):
        if direction not in ['inner', 'outer', None]:
            raise ValueError("tick direction=%s was requested, but tick "
                             "direction can only be one of "
                             "['outer', 'inner', None]." % direction)

        # left and right ticks and labels
        for y, ax in axes.iterrows():
            ax_ = ax.dropna()
            if len(ax_) and direction == 'inner':
                for i, a in enumerate(ax_):
                    if i == 0:  # first column
                        tl = a.yaxis.majorTicks[0].tick1line.get_markersize()
                        if a.position == 'diagonal' and len(ax_) == 1:
                            a.tick_params('y', left=False, labelleft=False,
                                          **kwargs)
                        else:
                            a.tick_params('y', left=True, labelleft=True,
                                          **kwargs)
                    elif a.position == 'diagonal':  # not first column
                        a.tick_params('y', direction='out', length=tl/2,
                                      left=True, labelleft=False, **kwargs)
                    elif ax_.iloc[i-1].position == 'diagonal':  # next to diag
                        a.tick_params('y', direction='in', length=tl/2,
                                      left=True, labelleft=False, **kwargs)
                    else:  # not diagonal and not first column
                        a.tick_params('y', direction='inout',
                                      left=True, labelleft=False, **kwargs)
            elif len(ax_) and direction == 'outer':  # no inner ticks
                for a in ax_.iloc[1:]:
                    a.tick_params('y', left=False, labelleft=False, **kwargs)
            elif len(ax_) and direction is None:  # no ticks at all
                for a in ax_:
                    a.tick_params('y', left=False, right=False,
                                  labelleft=False, labelright=False, **kwargs)

        # bottom and top ticks and labels
        for x, ax in axes.items():
            ax_ = ax.dropna()
            if len(ax_):
                if direction == 'inner':
                    for i, a in enumerate(ax_):
                        if i == len(ax_) - 1:  # bottom row
                            a.tick_params('x', bottom=True, labelbottom=True,
                                          **kwargs)
                        else:  # not bottom row
                            a.tick_params('x', direction='inout',
                                          bottom=True, labelbottom=False,
                                          **kwargs)
                            if a.position == 'diagonal':
                                a.twin.tick_params('x', direction='inout',
                                                   bottom=True,
                                                   labelbottom=False, **kwargs)
                elif direction == 'outer':  # no inner ticks
                    for a in ax_.iloc[:-1]:
                        a.tick_params('x', bottom=False, labelbottom=False,
                                      **kwargs)
                elif direction is None:  # no ticks at all
                    for a in ax_:
                        a.tick_params('x', bottom=False, top=False,
                                      labelbottom=False, labeltop=False,
                                      **kwargs)

    def tick_params(self, *args, **kwargs):
        """Apply :meth:`matplotlib.axes.Axes.tick_params` across all axes."""
        for y, rows in self.iterrows():
            for x, ax in rows.items():
                if isinstance(ax, Axes):
                    ax.tick_params(*args, **kwargs)

    def set_margins(self, m):
        """Apply :meth:`matplotlib.axes.Axes.set_xmargin` across all axes."""
        unique_params = list(np.unique(list(self.index) + list(self.columns)))
        for y, rows in self.iterrows():
            for x, ax in rows.items():
                if isinstance(ax, Axes):
                    if x in unique_params:
                        xmin, xmax = ax.get_xlim()
                        xdelta = xmax - xmin
                        ax.set_xlim(xmin - m * xdelta, xmax + m * xdelta)
                        unique_params.remove(x)
                    if y in unique_params:
                        ymin, ymax = ax.get_ylim()
                        ydelta = ymax - ymin
                        ax.set_ylim(ymin - m * ydelta, ymax + m * ydelta)
                        unique_params.remove(y)

    def axlines(self, params, lower=True, diagonal=True, upper=True, **kwargs):
        """Add vertical and horizontal lines across all axes.

        Parameters
        ----------
        params : dict(array_like)
            Dictionary of parameter labels and desired values.
            Can provide more than one value per label.
        lower, diagonal, upper : bool, default=True
            Whether to plot the lines on the lower, diagonal,
            and/or upper triangle plots.
        kwargs
            Any kwarg that can be passed to
            :meth:`matplotlib.axes.Axes.axvline` or
            :meth:`matplotlib.axes.Axes.axhline`.

        """
        positions = ['lower' if lower else None,
                     'diagonal' if diagonal else None,
                     'upper' if upper else None]
        for y, rows in self.iterrows():
            for x, ax in rows.items():
                if ax is not None and ax.position in positions:
                    if x in params:
                        for v in np.atleast_1d(params[x]):
                            if ax.position == 'diagonal':
                                ax.twin.axvline(v, **kwargs)
                            else:
                                ax.axvline(v, **kwargs)
                    if y in params and ax.position != 'diagonal':
                        for v in np.atleast_1d(params[y]):
                            ax.axhline(v, **kwargs)

    def axspans(self, params, lower=True, diagonal=True, upper=True, **kwargs):
        """Add vertical and horizontal spans across all axes.

        Parameters
        ----------
        params : dict(array_like(2-tuple))
            Dictionary of parameter labels and desired value tuples.
            Can provide more than one value tuple per label.
            Each value tuple provides the min and max value for an axis span.
        lower, diagonal, upper : bool, default=True
            Whether to plot the spans on the lower, diagonal,
            and/or upper triangle plots.
        kwargs
            Any kwarg that can be passed to
            :meth:`matplotlib.axes.Axes.axvspan` or
            :meth:`matplotlib.axes.Axes.axhspan`.

        """
        kwargs = normalize_kwargs(kwargs)
        positions = ['lower' if lower else None,
                     'diagonal' if diagonal else None,
                     'upper' if upper else None]
        for y, rows in self.iterrows():
            for x, ax in rows.items():
                if ax is not None and ax.position in positions:
                    if x in params:
                        for vmin, vmax in np.atleast_2d(params[x]):
                            if ax.position == 'diagonal':
                                ax.twin.axvspan(vmin, vmax, **kwargs)
                            else:
                                ax.axvspan(vmin, vmax, **kwargs)
                    if y in params and ax.position != 'diagonal':
                        for vmin, vmax in np.atleast_2d(params[y]):
                            ax.axhspan(vmin, vmax, **kwargs)

    def scatter(self, params, lower=True, upper=True, **kwargs):
        """Add scatter points across all axes.

        Parameters
        ----------
        params : dict(array_like)
            Dictionary of parameter labels and desired values.
            Can provide more than one value per label, but length has to
            match for all parameter labels.
        lower, upper : bool, default=True
            Whether to plot the spans on the lower and/or upper triangle plots.
        kwargs
            Any kwarg that can be passed to
            :meth:`matplotlib.axes.Axes.scatter`.

        """
        positions = ['lower' if lower else None,
                     'upper' if upper else None]
        zorder = kwargs.pop('zorder', None)
        for y, rows in self.iterrows():
            for x, ax in rows.items():
                if ax is not None and ax.position in positions:
                    if x in params and y in params:
                        z = max([z.get_zorder() for z in ax.artists +
                                 ax.collections + ax.lines + ax.patches] + [0])
                        z = z+1 if zorder is None else zorder
                        ax.scatter(params[x], params[y], zorder=z, **kwargs)


def make_1d_axes(params, ncol=None, labels=None, logx=None,
                 gridspec_kw=None, subplot_spec=None, **fig_kw):
    """Create a set of axes for plotting 1D marginalised posteriors.

    Parameters
    ----------
    params : list(str)
        names of parameters.

    ncol : int
        Number of columns of the subplot grid.
        Default: ceil(sqrt(num_params))

    labels : dict(str:str), optional
        Dictionary mapping params to plot labels.
        Default: params

    logx : list(str), optional
        List of parameters to be plotted on a log scale.

    gridspec_kw : dict, optional
        Dict with keywords passed to the :class:`matplotlib.gridspec.GridSpec`
        constructor used to create the grid the subplots are placed on.

    subplot_spec : :class:`matplotlib.gridspec.GridSpec`, default=None
        GridSpec instance to plot array as part of a subfigure.

    **fig_kw
        All additional keyword arguments are passed to the
        :func:`matplotlib.pyplot.figure` call.
        Or directly pass the figure to plot on via the keyword 'fig'.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        New or original (if supplied) figure object.

    axes : :class:`anesthetic.plot.AxesSeries`
        Pandas array of axes objects.

    """
    # TODO: remove this in version >= 2.1
    if 'tex' in fig_kw:
        raise NotImplementedError(
            "This is anesthetic 1.0 syntax. You need to update, e.g.\n"
            "make_1d_axes(..., tex=tex)     # anesthetic 1.0\n"
            "make_1d_axes(..., labels=tex)  # anesthetic 2.0"
            )
    fig = fig_kw.pop('fig') if 'fig' in fig_kw else plt.figure(**fig_kw)
    if np.array(params).ndim == 0:
        params = [params]
    axes = AxesSeries(index=params,
                      fig=fig,
                      ncol=ncol,
                      labels=labels,
                      logx=logx,
                      gridspec_kw=gridspec_kw,
                      subplot_spec=subplot_spec)
    if gridspec_kw is None:
        fig.tight_layout()
    return fig, axes


def make_2d_axes(params, labels=None, lower=True, diagonal=True, upper=True,
                 ticks='inner', logx=None, logy=None,
                 gridspec_kw=None, subplot_spec=None, **fig_kw):
    """Create a set of axes for plotting 2D marginalised posteriors.

    Parameters
    ----------
    params : lists of parameters
        Can be either:

        * ``list(str)`` if the x and y axes are the same
        * ``[list(str), list(str)]`` if the x and y axes are different

        Strings indicate the names of the parameters.

    labels : dict(str:str), optional
        Dictionary mapping params to plot labels.
        Default: params

    lower, diagonal, upper : logical, default=True
        Whether to create 2D marginalised plots above or below the
        diagonal, or to create a 1D marginalised plot on the diagonal.

    ticks : str, default='inner'
        Can be one of 'outer', 'inner', or None.

        * ``'outer'``: plot ticks only on the very left and very bottom.
        * ``'inner'``: plot ticks also in inner subplots.
        * ``None``: plot no ticks at all.

    logx, logy : list(str), optional
        Lists of parameters to be plotted on a log scale on the x-axis or
        y-axis, respectively.

    gridspec_kw : dict, optional
        Dict with keywords passed to the :class:`matplotlib.gridspec.GridSpec`
        constructor used to create the grid the subplots are placed on.

    subplot_spec : :class:`matplotlib.gridspec.GridSpec`, default=None
        GridSpec instance to plot array as part of a subfigure.

    **fig_kw
        All additional keyword arguments are passed to the
        :func:`matplotlib.pyplot.figure` call.
        Or directly pass the figure to plot on via the keyword 'fig'.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        New or original (if supplied) figure object.

    axes : :class:`anesthetic.plot.AxesDataFrame`
        Pandas array of axes objects.

    """
    # TODO: remove this in version >= 2.1
    if 'tex' in fig_kw:
        raise NotImplementedError(
            "This is anesthetic 1.0 syntax. You need to update, e.g.\n"
            "make_2d_axes(..., tex=tex)     # anesthetic 1.0\n"
            "make_2d_axes(..., labels=tex)  # anesthetic 2.0"
            )
    fig = fig_kw.pop('fig') if 'fig' in fig_kw else plt.figure(**fig_kw)

    if nest_level(params) == 2:
        xparams, yparams = params
    else:
        xparams = yparams = params
    axes = AxesDataFrame(index=yparams,
                         columns=xparams,
                         fig=fig,
                         lower=lower,
                         diagonal=diagonal,
                         upper=upper,
                         labels=labels,
                         ticks=ticks,
                         logx=logx,
                         logy=logy,
                         gridspec_kw=gridspec_kw,
                         subplot_spec=subplot_spec)
    fig.align_labels([a for a in axes.values.flatten() if isinstance(a, Axes)])
    return fig, axes


def fastkde_plot_1d(ax, data, *args, **kwargs):
    """Plot a 1d marginalised distribution.

    This functions as a wrapper around :meth:`matplotlib.axes.Axes.plot`, with
    a kernel density estimation (KDE) computation provided by the package
    fastkde in-between. All remaining keyword arguments are passed onwards.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axis object to plot on.

    data : np.array
        Uniformly weighted samples to generate kernel density estimator.

    xmin, xmax : float, default=None
        lower/upper prior bound

    levels : list
        Values at which to draw iso-probability lines.
        Optional,
        Default: [0.95, 0.68]

    q : int or float or tuple, default=5
        Quantile to determine the data range to be plotted.

        * ``0``: full data range, i.e. ``q=0`` --> quantile range (0, 1)
        * ``int``: q-sigma range, e.g. ``q=1`` --> quantile range (0.16, 0.84)
        * ``float``: percentile, e.g. ``q=0.8`` --> quantile range (0.1, 0.9)
        * ``tuple``: quantile range, e.g. (0.16, 0.84)

    facecolor : bool or string, default=False
        If set to True then the 1d plot will be shaded with the value of the
        ``color`` kwarg. Set to a string such as 'blue', 'k', 'r', 'C1' ect.
        to define the color of the shading directly.

    Returns
    -------
    lines : :class:`matplotlib.lines.Line2D`
        A list of line objects representing the plotted data (same as
        :meth:`matplotlib.axes.Axes.plot` command).

    """
    kwargs = normalize_kwargs(kwargs)
    if ax.get_xaxis().get_scale() == 'log':
        data = np.log10(data)
    xmin = kwargs.pop('xmin', None)
    xmax = kwargs.pop('xmax', None)
    levels = kwargs.pop('levels', [0.95, 0.68])
    density = kwargs.pop('density', False)

    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', (ax._get_lines.get_next_color()
                                 if cmap is None
                                 else plt.get_cmap(cmap)(0.68)))
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
        from anesthetic.kde import fastkde_1d
        x, p, xmin, xmax = fastkde_1d(data, xmin, xmax)
    except ImportError:
        raise ImportError("You need to install fastkde to use fastkde")
    p /= p.max()
    i = ((x > quantile(x, q[0], p)) & (x < quantile(x, q[-1], p)))

    if version.parse(np.__version__) >= version.parse("2.0.0"):
        trapezoid = np.trapezoid
    else:
        trapezoid = np.trapz
    area = trapezoid(x=x[i], y=p[i]) if density else 1
    if ax.get_xaxis().get_scale() == 'log':
        x = 10**x
    ans = ax.plot(x[i], p[i]/area, color=color, *args, **kwargs)

    if facecolor and facecolor not in [None, 'None', 'none']:
        if facecolor is True:
            facecolor = color
        levels = iso_probability_contours(p[i], contours=levels)
        cmap = basic_cmap(facecolor)
        fill = []
        for level in levels:
            fill.append(ax.fill_between(x[i], p[i], where=p[i] >= level,
                        color=cmap(level), edgecolor=edgecolor))

        return ans, fill

    return ans


def kde_plot_1d(ax, data, *args, **kwargs):
    """Plot a 1d marginalised distribution.

    This functions as a wrapper around :meth:`matplotlib.axes.Axes.plot`, with
    a kernel density estimation computation provided by
    :class:`scipy.stats.gaussian_kde` in-between. All remaining keyword
    arguments are passed onwards.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axis object to plot on.

    data : np.array
        Samples to generate kernel density estimator.

    weights : np.array, optional
        Sample weights.

    ncompress : int, str, default=False
        Degree of compression.

        * If ``False``: no compression.
        * If ``True``: compresses to the channel capacity, equivalent to
          ``ncompress='entropy'``.
        * If ``int``: desired number of samples after compression.
        * If ``str``: determine number from the Huggins-Roy family of
          effective samples in :func:`anesthetic.utils.neff`
          with ``beta=ncompress``.

    nplot_1d : int, default=100
        Number of plotting points to use.

    levels : list
        Values at which to draw iso-probability lines.
        Default: [0.95, 0.68]

    q : int or float or tuple, default=5
        Quantile to determine the data range to be plotted.

        * ``0``: full data range, i.e. ``q=0`` --> quantile range (0, 1)
        * ``int``: q-sigma range, e.g. ``q=1`` --> quantile range (0.16, 0.84)
        * ``float``: percentile, e.g. ``q=0.8`` --> quantile range (0.1, 0.9)
        * ``tuple``: quantile range, e.g. (0.16, 0.84)

    facecolor : bool or string, default=False
        If set to True then the 1d plot will be shaded with the value of the
        ``color`` kwarg. Set to a string such as 'blue', 'k', 'r', 'C1' ect.
        to define the color of the shading directly.

    bw_method : str, scalar or callable, optional
        Forwarded to :class:`scipy.stats.gaussian_kde`.

    bw_scale : float, default=1
        Scales the bandwidth relative to the automatically computed one by
        :class:`scipy.stats.gaussian_kde`. A value greater 1 will smooth more,
        a value smaller 1 will smooth less.

    Returns
    -------
    lines : :class:`matplotlib.lines.Line2D`
        A list of line objects representing the plotted data (same as
        :meth:`matplotlib.axes.Axes.plot` command).

    """
    kwargs = normalize_kwargs(kwargs)
    weights = kwargs.pop('weights', None)
    if weights is not None:
        data = data[weights != 0]
        weights = weights[weights != 0]
    if ax.get_xaxis().get_scale() == 'log':
        data = np.log10(data)

    ncompress = kwargs.pop('ncompress', False)
    nplot = kwargs.pop('nplot_1d', 100)
    bw_method = kwargs.pop('bw_method', None)
    bw_scale = kwargs.pop('bw_scale', 1)
    order = kwargs.pop('order', 1)
    levels = kwargs.pop('levels', [0.95, 0.68])
    density = kwargs.pop('density', False)

    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', (ax._get_lines.get_next_color()
                                 if cmap is None
                                 else plt.get_cmap(cmap)(0.68)))
    facecolor = kwargs.pop('facecolor', False)
    if 'edgecolor' in kwargs:
        edgecolor = kwargs.pop('edgecolor')
        if edgecolor:
            color = edgecolor
    else:
        edgecolor = color

    if np.var(data) <= 0:
        noise = _plot_window(ax, 'x') * 1e-5
        data = data.copy() + noise * np.random.normal(size=data.size)

    q = kwargs.pop('q', 5)
    q = quantile_plot_interval(q=q)
    xmin = quantile(data, q[0], weights)
    xmax = quantile(data, q[-1], weights)
    x = np.linspace(xmin, xmax, nplot)
    for edge, direction in [(data.min(), -np.inf), (data.max(), np.inf)]:
        if xmin <= edge <= xmax:
            x = np.union1d(x, [np.nextafter(edge, direction)])

    data_compressed, w = sample_compression_1d(data, weights, ncompress)
    kde = gaussian_kde(data_compressed, weights=w, bw_method=bw_method)
    kde.set_bandwidth(bw_method=kde.factor * bw_scale)

    p = boundary_correction_1d(kde, x, order=order,
                               xmin=data.min(), xmax=data.max())
    p /= p.max()
    if version.parse(np.__version__) >= version.parse("2.0.0"):
        trapezoid = np.trapezoid
    else:
        trapezoid = np.trapz
    area = trapezoid(x=x, y=p) if density else 1
    if ax.get_xaxis().get_scale() == 'log':
        x = 10**x
    ans = ax.plot(x, p/area, color=color, *args, **kwargs)

    if facecolor and facecolor not in [None, 'None', 'none']:
        if facecolor is True:
            facecolor = color
        levels = iso_probability_contours(p, contours=levels)
        cmap = basic_cmap(facecolor)
        fill = []
        for level in levels:
            fill.append(ax.fill_between(x, p, where=p >= level,
                        color=cmap(level), edgecolor=edgecolor))

        ans = ans, fill

    if density:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(0, 1.1)

    return ans


def hist_plot_1d(ax, data, *args, **kwargs):
    """Plot a 1d histogram.

    This functions is a wrapper around :meth:`matplotlib.axes.Axes.hist`. All
    remaining keyword arguments are passed onwards.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axis object to plot on.

    data : np.array
        Samples to generate histogram from

    weights : np.array, optional
        Sample weights.

    q : int or float or tuple, default=5
        Quantile to determine the data range to be plotted.

        * ``0``: full data range, i.e. ``q=0`` --> quantile range (0, 1)
        * ``int``: q-sigma range, e.g. ``q=1`` --> quantile range (0.16, 0.84)
        * ``float``: percentile, e.g. ``q=0.8`` --> quantile range (0.1, 0.9)
        * ``tuple``: quantile range, e.g. (0.16, 0.84)

    Returns
    -------
    patches : list or list of lists
        Silent list of individual patches used to create the histogram
        or list of such list if multiple input datasets.

    Other Parameters
    ----------------
    **kwargs : :meth:`matplotlib.axes.Axes.hist` properties

    """
    kwargs = normalize_kwargs(kwargs)
    weights = kwargs.pop('weights', None)
    bins = kwargs.pop('bins', 'fd')
    histtype = kwargs.pop('histtype', 'bar')
    density = kwargs.get('density', False)

    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', (ax._get_lines.get_next_color()
                                 if cmap is None
                                 else plt.get_cmap(cmap)(0.68)))

    q = kwargs.pop('q', 5)
    q = quantile_plot_interval(q=q)
    if ax.get_xaxis().get_scale() == 'log':
        data = np.log10(data)
    xmin = quantile(data, q[0], weights)
    xmax = quantile(data, q[-1], weights)
    if 'range' in kwargs and ax.get_xaxis().get_scale() == 'log':
        range = kwargs.pop('range')
        if range is not None:
            range = (np.log10(range[0]), np.log10(range[1]))
        else:
            range = (data.min(), data.max())
    else:
        range = kwargs.pop('range', (xmin, xmax))
    if isinstance(bins, (int, str)):
        if isinstance(bins, int):
            bins = np.linspace(range[0], range[1], bins+1)
        elif isinstance(bins, str) and bins in ['fd', 'scott', 'sqrt']:
            bins = histogram_bin_edges(data,
                                       weights=weights,
                                       bins=bins,
                                       beta=kwargs.pop('beta', 'equal'),
                                       range=range)
        if ax.get_xaxis().get_scale() == 'log':
            bins = 10 ** bins
    if ax.get_xaxis().get_scale() == 'log':
        data = 10**data
        range = (10**range[0], 10**range[1])
    if isinstance(bins, str) and bins in ['knuth', 'freedman', 'blocks']:
        raise ValueError("The astropy strings 'knuth', 'freedman', and "
                         "'blocks' are no longer supported. Please use the"
                         "similar 'fd', 'scott', or 'sqrt' from now on.")
    else:
        h, edges, bars = ax.hist(data, weights=weights, bins=bins,
                                 range=range, histtype=histtype,
                                 color=color, *args, **kwargs)

    if histtype == 'bar' and not density:
        for b in bars:
            b.set_height(b.get_height() / h.max())
    elif (histtype == 'step' or histtype == 'stepfilled') and not density:
        trans = Affine2D().scale(sx=1, sy=1./h.max()) + ax.transData
        bars[0].set_transform(trans)

    if not density:
        ax.set_ylim(0, 1.1)
    return h, edges, bars


def fastkde_contour_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot a 2d marginalised distribution as contours.

    This functions as a wrapper around :meth:`matplotlib.axes.Axes.contour`,
    and :meth:`matplotlib.axes.Axes.contourf` with a kernel density estimation
    (KDE) computation in-between. All remaining keyword arguments are passed
    onwards to both functions.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axis object to plot on.

    data_x, data_y : np.array
        The x and y coordinates of uniformly weighted samples to generate
        kernel density estimator.

    levels : list
        Amount of mass within each iso-probability contour.
        Has to be ordered from outermost to innermost contour.
        Default: [0.95, 0.68]

    xmin, xmax, ymin, ymax : float, default=None
        The lower/upper prior bounds in x/y coordinates.

    Returns
    -------
    c : :class:`matplotlib.contour.QuadContourSet`
        A set of contourlines or filled regions.

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
    if ax.get_xaxis().get_scale() == 'log':
        data_x = np.log10(data_x)
        xmin = None if xmin is None else np.log10(xmin)
        xmax = None if xmax is None else np.log10(xmax)
    if ax.get_yaxis().get_scale() == 'log':
        data_y = np.log10(data_y)
        ymin = None if ymin is None else np.log10(ymin)
        ymax = None if ymax is None else np.log10(ymax)
    label = kwargs.pop('label', None)
    zorder = kwargs.pop('zorder', 1)
    levels = kwargs.pop('levels', [0.95, 0.68])

    color = kwargs.pop('color', ax._get_lines.get_next_color())
    facecolor = kwargs.pop('facecolor', True)
    edgecolor = kwargs.pop('edgecolor', None)
    cmap = kwargs.pop('cmap', None)
    facecolor, edgecolor, cmap = set_colors(c=color, fc=facecolor,
                                            ec=edgecolor, cmap=cmap)

    kwargs.pop('q', None)

    try:
        from anesthetic.kde import fastkde_2d
        x, y, pdf, xmin, xmax, ymin, ymax = fastkde_2d(data_x, data_y,
                                                       xmin=xmin, xmax=xmax,
                                                       ymin=ymin, ymax=ymax)
    except ImportError:
        raise ImportError("You need to install fastkde to use fastkde")

    levels = iso_probability_contours(pdf, contours=levels) + [pdf.max()]

    i = (pdf >= levels[0]*0.5).any(axis=0)
    j = (pdf >= levels[0]*0.5).any(axis=1)

    if ax.get_xaxis().get_scale() == 'log':
        x = 10**x
    if ax.get_yaxis().get_scale() == 'log':
        y = 10**y

    if facecolor not in [None, 'None', 'none']:
        linewidths = kwargs.pop('linewidths', 0.5)
        contf = ax.contourf(x[i], y[j], pdf[np.ix_(j, i)], levels, cmap=cmap,
                            zorder=zorder, vmin=0, vmax=pdf.max(),
                            *args, **kwargs)
        contf.set_cmap(cmap)
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

    return contf, cont


def kde_contour_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot a 2d marginalised distribution as contours.

    This functions as a wrapper around :meth:`matplotlib.axes.Axes.contour`
    and :meth:`matplotlib.axes.Axes.contourf` with a kernel density
    estimation (KDE) computation provided by :class:`scipy.stats.gaussian_kde`
    in-between. All remaining keyword arguments are passed onwards to both
    functions.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axis object to plot on.

    data_x, data_y : np.array
        The x and y coordinates of uniformly weighted samples to generate
        kernel density estimator.

    weights : np.array, optional
        Sample weights.

    levels : list, optional
        Amount of mass within each iso-probability contour.
        Has to be ordered from outermost to innermost contour.
        Default: [0.95, 0.68]

    ncompress : int, str, default='equal'
        Degree of compression.

        * If ``int``: desired number of samples after compression.
        * If ``False``: no compression.
        * If ``True``: compresses to the channel capacity, equivalent to
          ``ncompress='entropy'``.
        * If ``str``: determine number from the Huggins-Roy family of
          effective samples in :func:`anesthetic.utils.neff`
          with ``beta=ncompress``.

    nplot_2d : int, default=1000
        Number of plotting points to use.

    bw_method : str, scalar or callable, optional
        Forwarded to :class:`scipy.stats.gaussian_kde`.

    bw_scale : float, default=1
        Scales the bandwidth relative to the automatically computed one by
        :class:`scipy.stats.gaussian_kde`. A value greater 1 will smooth more,
        a value smaller 1 will smooth less.

    grid_angle : float or (float, float), optional
        Manual orientation of the plotting grid, in degrees measured
        counter-clockwise from the +x axis. If set, forces use of the
        basis-aligned grid regardless of the measured correlation.

        * ``None`` (default): grid axes are derived from the eigenvectors of
          the sample covariance when the correlation is high, else the grid is
          axis-aligned.
        * ``float``: direction of the major grid axis. The minor grid
          axis is placed perpendicular to it.
        * ``(major, minor)``: both grid axis directions specified explicitly.
          If the two are not perpendicular, grid cells become parallelograms
          rather than rectangles.

    Returns
    -------
    c : :class:`matplotlib.contour.QuadContourSet`
        A set of contourlines or filled regions.

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
    if ax.get_xaxis().get_scale() == 'log':
        data_x = np.log10(data_x)
    if ax.get_yaxis().get_scale() == 'log':
        data_y = np.log10(data_y)

    ncompress = kwargs.pop('ncompress', 'equal')
    nplot = kwargs.pop('nplot_2d', 1000)
    grid_angle = kwargs.pop('grid_angle', None)
    bw_method = kwargs.pop('bw_method', None)
    bw_scale = kwargs.pop('bw_scale', 1)
    order = kwargs.pop('order', None)
    label = kwargs.pop('label', None)
    zorder = kwargs.pop('zorder', 1)
    levels = kwargs.pop('levels', [0.95, 0.68])

    color = kwargs.pop('color', ax._get_lines.get_next_color())
    facecolor = kwargs.pop('facecolor', True)
    edgecolor = kwargs.pop('edgecolor', None)
    cmap = kwargs.pop('cmap', None)
    facecolor, edgecolor, cmap = set_colors(c=color, fc=facecolor,
                                            ec=edgecolor, cmap=cmap)

    # Regularise degenerate input (collinear or constant data) by injecting
    # small noise so that the covariance is positive-definite for Cholesky
    # decomposition in scaled_triangulation and gaussian_kde.
    cov = np.cov(data_x, data_y, aweights=weights)
    (var_x, cov_xy), (cov_yx, var_y) = cov
    corr = 0 if var_x <= 0 or var_y <= 0 else abs(cov_xy)/np.sqrt(var_x*var_y)
    eig = None
    if var_x <= 0 or var_y <= 0 or corr > 1 - np.sqrt(np.finfo(float).eps):
        eig = np.linalg.eigh(cov)
        evals, evecs = eig
        noise = np.sqrt(max(0, evals[0])) * 1e-3
        noise += np.sqrt(max(0, evals[1])) * 1e-3 if abs(cov_xy) > 0 else 0
        noise += _plot_window(ax, 'x') * 1e-3 if var_x <= 0 else 0
        noise += _plot_window(ax, 'y') * 1e-3 if var_y <= 0 else 0
        noise = noise * np.random.normal(size=data_x.size)
        data_x = data_x.copy() + noise * evecs[0, 0]
        data_y = data_y.copy() + noise * evecs[1, 0]
        cov = np.cov(data_x, data_y, aweights=weights)

    q = kwargs.pop('q', 5)
    q = quantile_plot_interval(q=q)
    xmin = quantile(data_x, q[0], weights)
    xmax = quantile(data_x, q[-1], weights)
    ymin = quantile(data_y, q[0], weights)
    ymax = quantile(data_y, q[-1], weights)
    ngrid = int(np.sqrt(nplot))
    if corr > 0.99 or grid_angle is not None:
        if grid_angle is None and eig is None:
            eig = np.linalg.eigh(cov)
        X, Y, n_vec, n_min, n_max = _basis_aligned_grid(
            data_x, data_y, eig, ngrid, xmin, xmax, ymin, ymax, grid_angle
        )
        n_kwargs = dict(n_vec=n_vec, nmin=n_min, nmax=n_max)
    else:
        x = np.linspace(xmin, xmax, ngrid)
        y = np.linspace(ymin, ymax, ngrid)
        for edge, direction in [(data_x.min(), -np.inf),
                                (data_x.max(), np.inf)]:
            if xmin <= edge <= xmax:
                x = np.union1d(x, [np.nextafter(edge, direction)])
        for edge, direction in [(data_y.min(), -np.inf),
                                (data_y.max(), np.inf)]:
            if ymin <= edge <= ymax:
                y = np.union1d(y, [np.nextafter(edge, direction)])
        X, Y = np.meshgrid(x, y)
        n_kwargs = {}
    x_grid, y_grid = X.ravel(), Y.ravel()

    tri, w = triangular_sample_compression_2d(data_x, data_y, cov,
                                              weights, ncompress)
    kde = gaussian_kde([tri.x, tri.y], weights=w, bw_method=bw_method)
    kde.set_bandwidth(bw_method=kde.factor * bw_scale)

    # Evaluate boundary-corrected KDE on grid + sample vertices in one pass.
    # Grid values are used for plotting; sample values for computing
    # iso-probability levels independently of the plotting window.
    # Subsample vertices for level computation to avoid O(n_samples^2) cost.
    n_samp = min(len(tri.x), max(1000, int(100 / (1 - max(levels)))))
    idx = np.random.choice(len(tri.x), n_samp, replace=False)
    x_samp, y_samp, w_samp = tri.x[idx], tri.y[idx], w[idx]

    x_all = np.concatenate([x_grid, x_samp])
    y_all = np.concatenate([y_grid, y_samp])
    p_all = boundary_correction_2d(kde, x_all, y_all, order=order,
                                   xmin=data_x.min(), xmax=data_x.max(),
                                   ymin=data_y.min(), ymax=data_y.max(),
                                   **n_kwargs)
    P_plot = p_all[:-n_samp].reshape(X.shape)
    p_samp = p_all[-n_samp:]
    levels = iso_probability_contours_from_samples(p_samp,
                                                   contours=levels,
                                                   weights=w_samp)
    vmax = max(P_plot.max(), p_samp.max())
    levels = levels + [vmax]
    if ax.get_xaxis().get_scale() == 'log':
        X = 10**X
    if ax.get_yaxis().get_scale() == 'log':
        Y = 10**Y

    if facecolor not in [None, 'None', 'none']:
        linewidths = kwargs.pop('linewidths', 0.5)
        contf = ax.contourf(X, Y, P_plot, levels=levels, cmap=cmap,
                            zorder=zorder, vmin=0, vmax=vmax,
                            *args, **kwargs)
        contf.set_cmap(cmap)
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

    vmin, vmax = match_contour_to_contourf(levels, vmin=0, vmax=vmax)
    cont = ax.contour(X, Y, P_plot, levels=levels, zorder=zorder,
                      vmin=vmin, vmax=vmax, linewidths=linewidths,
                      colors=edgecolor, cmap=cmap, *args, **kwargs)

    return contf, cont


def hist_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot a 2d marginalised distribution as a histogram.

    This functions as a wrapper around :meth:`matplotlib.axes.Axes.hist2d`.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axis object to plot on.

    data_x, data_y : np.array
        The x and y coordinates of uniformly weighted samples to generate a
        two-dimensional histogram.

    levels : list, default=None
        Shade iso-probability contours containing these levels of probability
        mass. If None defaults to usual :meth:`matplotlib.axes.Axes.hist2d`
        colouring.

    q : int or float or tuple, default=5
        Quantile to determine the data range to be plotted.

        * ``0``: full data range, i.e. ``q=0`` --> quantile range (0, 1)
        * ``int``: q-sigma range, e.g. ``q=1`` --> quantile range (0.16, 0.84)
        * ``float``: percentile, e.g. ``q=0.8`` --> quantile range (0.1, 0.9)
        * ``tuple``: quantile range, e.g. (0.16, 0.84)

    Returns
    -------
    c : :class:`matplotlib.collections.QuadMesh`
        A set of colors.

    """
    kwargs = normalize_kwargs(kwargs)
    weights = kwargs.pop('weights', None)
    if ax.get_xaxis().get_scale() == 'log':
        data_x = np.log10(data_x)
    if ax.get_yaxis().get_scale() == 'log':
        data_y = np.log10(data_y)

    vmin = kwargs.pop('vmin', 0)
    label = kwargs.pop('label', None)
    levels = kwargs.pop('levels', None)

    color = kwargs.pop('color', ax._get_lines.get_next_color())
    cmap = kwargs.pop('cmap', basic_cmap(color))

    q = kwargs.pop('q', 5)
    q = quantile_plot_interval(q=q)
    xmin = quantile(data_x, q[0], weights)
    xmax = quantile(data_x, q[-1], weights)
    ymin = quantile(data_y, q[0], weights)
    ymax = quantile(data_y, q[-1], weights)
    rge = kwargs.pop('range', ((xmin, xmax), (ymin, ymax)))

    bins = kwargs.pop('bins', 10)
    density = kwargs.pop('density', False)
    cmin = kwargs.pop('cmin', None)
    cmax = kwargs.pop('cmax', None)
    pdf, x, y = np.histogram2d(data_x, data_y, bins, rge,
                               density, weights)
    if levels is not None:
        levels = iso_probability_contours(pdf, levels) + [pdf.max()]
        pdf = np.digitize(pdf, levels, right=True)
        pdf = np.array(levels)[pdf]
        pdf = np.ma.masked_array(pdf, pdf < levels[1])
        if cmin is not None:
            pdf[pdf < cmin] = np.ma.masked
        if cmax is not None:
            pdf[pdf > cmax] = np.ma.masked
    snap = kwargs.pop('snap', True)
    if ax.get_xaxis().get_scale() == 'log':
        x = 10**x
    if ax.get_yaxis().get_scale() == 'log':
        y = 10**y
    image = ax.pcolormesh(x, y, pdf.T, cmap=cmap, vmin=vmin, snap=snap,
                          *args, **kwargs)

    ax.add_patch(plt.Rectangle((0, 0), 0, 0, fc=cmap(0.999), ec=cmap(0.32),
                               lw=2, label=label))

    return image


def scatter_plot_2d(ax, data_x, data_y, *args, **kwargs):
    """Plot samples from a 2d marginalised distribution.

    This functions as a wrapper around :meth:`matplotlib.axes.Axes.plot`,
    enforcing any prior bounds. All remaining keyword arguments are passed
    onwards.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        axis object to plot on

    data_x, data_y : np.array
        x and y coordinates of uniformly weighted samples to plot.

    ncompress : int, str, default='equal'
        Degree of compression.

        * If ``int``: desired number of samples after compression.
        * If ``False``: no compression.
        * If ``True``: compresses to the channel capacity, equivalent to
          ``ncompress='entropy'``.
        * If ``str``: determine number from the Huggins-Roy family of
          effective samples in :func:`anesthetic.utils.neff`
          with ``beta=ncompress``.

    Returns
    -------
    lines : :class:`matplotlib.lines.Line2D`
        A list of line objects representing the plotted data (same as
        :meth:`matplotlib.axes.Axes.plot` command).

    """
    kwargs = normalize_kwargs(
        kwargs,
        alias_mapping=dict(lw=['linewidth', 'linewidths'],
                           ls=['linestyle', 'linestyles'],
                           color=['c'],
                           mfc=['fc', 'facecolor'],
                           mec=['ec', 'edgecolor'],
                           cmap=['colormap']),
        drop=['ls', 'lw']
    )
    kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)

    markersize = kwargs.pop('markersize', 1)
    cmap = kwargs.pop('cmap', None)
    color = kwargs.pop('color', (ax._get_lines.get_next_color()
                                 if cmap is None else cmap(0.68)))

    weights = kwargs.pop('weights', None)
    q = kwargs.pop('q', 5)
    q = quantile_plot_interval(q=q)
    xmin = quantile(data_x, q[0], weights)
    xmax = quantile(data_x, q[-1], weights)
    ymin = quantile(data_y, q[0], weights)
    ymax = quantile(data_y, q[-1], weights)
    mask = ((data_x >= xmin) & (data_x <= xmax) &
            (data_y >= ymin) & (data_y <= ymax))

    points = ax.plot(data_x[mask], data_y[mask], 'o',
                     color=color, markersize=markersize, *args, **kwargs)
    return points


def basic_cmap(color):
    """Construct basic colormap a single color."""
    return LinearSegmentedColormap.from_list(str(color), ['#ffffff', color])


def quantile_plot_interval(q):
    """Interpret quantile ``q`` input to quantile plot range tuple."""
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

    Works the same way as :func:`matplotlib.cbook.normalize_kwargs`, but
    additionally allows to drop kwargs.
    """
    drop = [] if drop is None else drop
    if alias_mapping is None:
        alias_mapping = dict(linewidth=['lw'],
                             linestyle=['ls'],
                             color=['c'],
                             facecolor=['fc'],
                             edgecolor=['ec'],
                             cmap=['colormap'])
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


def _plot_window(ax, axis):
    """Return the displayed width or height of an axis.

    Uses axis limits from ``viewLim`` when explicitly set via
    ``set_xlim``/``set_ylim``, or from ``dataLim`` when prior data has been
    plotted. Raises ``ValueError`` if neither is available.
    """
    if axis == 'x':
        has_limits = not ax.get_autoscalex_on()
        interval = ax.viewLim.intervalx if has_limits else ax.dataLim.intervalx
        scale = ax.get_xscale()
    else:
        has_limits = not ax.get_autoscaley_on()
        interval = ax.viewLim.intervaly if has_limits else ax.dataLim.intervaly
        scale = ax.get_yscale()
    if np.isfinite(interval).all():
        if scale == 'log':
            interval = np.log10(interval)
        return (interval[1] - interval[0])
    raise ValueError(
        f"Cannot plot KDE contours: the {axis}-axis variable has zero "
        f"variance and no axis limits are set. Call `ax.set_{axis}lim(...)` "
        f"on the " f"corresponding axis before (and not again after!) "
        f"plotting to define the display range."
    )


def _basis_aligned_grid(data_x, data_y, eig, ngrid,
                        xmin, xmax, ymin, ymax, grid_angle=None):
    """Return a basis-aligned grid clipped to the axis-aligned plot window.

    By default the minor and major grid directions are the eigenvectors
    of the sample covariance (ascending eigenvalues). If ``grid_angle``
    is provided, it overrides the covariance-derived axes: a scalar sets
    the major axis direction (minor is placed perpendicular), a pair
    ``(major, minor)`` sets both explicitly. Angles are in degrees
    counter-clockwise from the +x axis. When the two axes are not
    perpendicular the grid cells are parallelograms.
    """
    if grid_angle is None:
        _, evecs = eig
        u_vec = evecs[:, 0].copy()  # minor axis
        v_vec = evecs[:, 1].copy()  # major axis
    else:
        if np.ndim(grid_angle) == 0:
            major, minor = grid_angle, grid_angle + 90.0
        else:
            major, minor = grid_angle
        v_vec = np.array([np.cos(np.deg2rad(major)),
                          np.sin(np.deg2rad(major))])
        u_vec = np.array([np.cos(np.deg2rad(minor)),
                          np.sin(np.deg2rad(minor))])
        if abs(u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]) < 1e-10:
            raise ValueError(f"grid_angle major ({major}) and minor ({minor}) "
                             f"axes are (near-)parallel; cannot build a grid.")

    # Snap near-zero components (e.g. cos(90°) ≈ 6e-17) to zero so
    # axis-aligned basis vectors are exact.
    v_vec[np.abs(v_vec) < 1e-12] = 0.0
    u_vec[np.abs(u_vec) < 1e-12] = 0.0

    # Eigenvectors are sign-degenerate, and angles that differ by 180 degrees
    # describe the same grid axis. Point v towards +x, or towards +y when it
    # is vertical, so the grid orientation is reproducible.
    if v_vec[0] < 0 or (v_vec[0] == 0 and v_vec[1] < 0):
        v_vec *= -1
    # The minor/u axis is sign-degenerate as well. Point u to the left of v,
    # matching the scalar grid_angle convention where minor = major + 90 deg.
    if v_vec[0] * u_vec[1] - v_vec[1] * u_vec[0] < 0:
        u_vec *= -1
    # Use the same deterministic orientation for n, the normal to v. It points
    # to the same side of v as u, so increasing u also increases n.
    n_vec = np.array([-v_vec[1], v_vec[0]])

    M = np.column_stack([u_vec, v_vec])
    uv_data = np.linalg.solve(M, np.vstack([data_x, data_y]))
    u = uv_data[0]
    xy_corners = np.array([[xmin, ymin],
                           [xmin, ymax],
                           [xmax, ymin],
                           [xmax, ymax]])
    uv_corners = np.linalg.solve(M, xy_corners.T).T
    umin_corner = uv_corners[:, 0].min()
    umax_corner = uv_corners[:, 0].max()
    umin = max(umin_corner, u.min())
    umax = min(umax_corner, u.max())
    vmin = uv_corners[:, 1].min()
    vmax = uv_corners[:, 1].max()
    u_grid = np.linspace(umin, umax, ngrid)
    # Unit normal to v_vec gives a true axis perpendicular to the rotated
    # rows. For orthonormal bases this coincides with u_vec; for sheared
    # bases (e.g. grid_angle=(45, 0)) they differ. Project the data
    # onto this normal so `boundary_correction_2d` can apply a
    # separable Jones-style 1D correction along the rotated direction.
    n_proj = n_vec[0] * data_x + n_vec[1] * data_y
    # Add one row just outside the data's u extents so density can be
    # forced to zero there, giving cleanly closed contours along rotated edges.
    n_scale = max(1, abs(n_proj.min()), abs(n_proj.max()))
    u_step = 16 * np.finfo(u_grid.dtype).eps * n_scale / (n_vec @ u_vec)
    extra_edges = [edge + direction * u_step
                   for edge, direction in [(u.min(), -1), (u.max(), +1)]
                   if umin <= edge <= umax]
    u_grid = np.union1d(u_grid, extra_edges)
    vlos = np.full_like(u_grid, vmin)
    vhis = np.full_like(u_grid, vmax)
    for uj, vj, zmin, zmax in [(u_vec[0], v_vec[0], xmin, xmax),
                               (u_vec[1], v_vec[1], ymin, ymax)]:
        if vj == 0:
            continue
        a = (zmin - uj * u_grid) / vj
        b = (zmax - uj * u_grid) / vj
        vlos = np.maximum(vlos, np.minimum(a, b))
        vhis = np.minimum(vhis, np.maximum(a, b))

    V = np.array([np.linspace(vlo, vhi, ngrid)
                  for vlo, vhi in zip(vlos, vhis)])
    U = np.broadcast_to(u_grid[:, None], V.shape)
    X = u_vec[0] * U + v_vec[0] * V
    Y = u_vec[1] * U + v_vec[1] * V

    # Reconstructing X/Y from U/V can move algebraic boundary points by one
    # ulp; snap the core grid back to the precise bounds.
    for Z, zmin, zmax in [(X, xmin, xmax), (Y, ymin, ymax)]:
        atol = 8 * np.finfo(Z.dtype).eps * max(1, abs(zmin), abs(zmax))
        Z[np.isclose(Z, zmin, rtol=0, atol=atol)] = zmin
        Z[np.isclose(Z, zmax, rtol=0, atol=atol)] = zmax

    # Exand the grid at the boundaries for cleanly closed contours.
    if xmin <= data_x.min() or ((ymin <= data_y.min() and v_vec[1] >= 0) or
                                (ymax >= data_y.max() and v_vec[1] <= 0)):
        X = np.column_stack([np.nextafter(X[:, 0], X[:, 0] - v_vec[0]), X])
        Y = np.column_stack([np.nextafter(Y[:, 0], Y[:, 0] - v_vec[1]), Y])
    if xmax >= data_x.max() or ((ymax >= data_y.max() and v_vec[1] >= 0) or
                                (ymin <= data_y.min() and v_vec[1] <= 0)):
        X = np.column_stack([X, np.nextafter(X[:, -1], X[:, -1] + v_vec[0])])
        Y = np.column_stack([Y, np.nextafter(Y[:, -1], Y[:, -1] + v_vec[1])])

    return X, Y, n_vec, n_proj.min(), n_proj.max()
