from __future__ import annotations
import pandas
from pandas.plotting import PlotAccessor as _PlotAccessor
from anesthetic.plot import make_1d_axes, make_2d_axes


def _process_docstring(doc):
    i = doc.find('    ax')
    e = (
        "    - 'hist_1d' : 1d histogram\n"
        "    - 'kde_1d' : 1d Kernel Density Estimation plot\n"
        "    - 'fastkde_1d' : 1d Kernel Density Estimation plot"
        "                     with fastkde package\n"
        "    - 'hist_2d' : 2d histogram (DataFrame only)\n"
        "    - 'kde_2d' : 2d Kernel Density Estimation plot (DataFrame only)\n"
        "    - 'fastkde_2d' : 2d Kernel Density Estimation plot"
        "                     with fastkde package (DataFrame only)\n"
        "    - 'scatter_2d' : 2d scatter plot (DataFrame only)\n"
        )
    return doc[:i] + e + doc[i:]


class PlotAccessor(_PlotAccessor):
    __doc__ = _process_docstring(_PlotAccessor.__doc__)
    _common_kinds = _PlotAccessor._common_kinds \
        + ("hist_1d", "kde_1d", "fastkde_1d")
    _series_kinds = _PlotAccessor._series_kinds + ()
    _dataframe_kinds = _PlotAccessor._dataframe_kinds \
        + ("hist_2d", "kde_2d", "fastkde_2d", "scatter_2d")
    _all_kinds = _common_kinds + _series_kinds + _dataframe_kinds

    def hist_1d(self, **kwargs) -> PlotAccessor:
        return self(kind="hist_1d", **kwargs)

    def kde_1d(self, **kwargs) -> PlotAccessor:
        return self(kind="kde_1d", **kwargs)

    def fastkde_1d(self, **kwargs) -> PlotAccessor:
        return self(kind="fastkde_1d", **kwargs)

    def kde_2d(self, x, y, **kwargs) -> PlotAccessor:
        return self(kind="kde_2d", x=x, y=y, **kwargs)

    def fastkde_2d(self, x, y, **kwargs) -> PlotAccessor:
        return self(kind="fastkde_2d", x=x, y=y, **kwargs)

    def hist_2d(self, x, y, **kwargs) -> PlotAccessor:
        return self(kind="hist_2d", x=x, y=y, **kwargs)

    def scatter_2d(self, x, y, **kwargs) -> PlotAccessor:
        return self(kind="scatter_2d", x=x, y=y, **kwargs)


def plot_1d(self, axes, *args, **kwargs):
    """Create an array of 1D plots.

    Parameters
    ----------
    axes: plotting axes
        Can be:
            - list(str) or str
            - pandas.Series(matplotlib.axes.Axes)
        If a pandas.Series is provided as an existing set of axes, then
        this is used for creating the plot. Otherwise a new set of axes are
        created using the list or lists of strings.

    Returns
    -------
    fig: matplotlib.figure.Figure
        New or original (if supplied) figure object

    axes: pandas.Series of matplotlib.axes.Axes
        Pandas array of axes objects

    """
    self._set_automatic_limits()

    if not isinstance(axes, pandas.Series):
        fig, axes = make_1d_axes(axes, tex=self.tex)
    else:
        fig = axes.bfill().to_numpy().flatten()[0].figure

    kwargs['kind'] = kwargs.get('kind', 'kde_1d')
    kwargs['label'] = kwargs.get('label', self.label)

    for x, ax in axes.iteritems():
        if x in self:
            self[x].plot(ax=ax, *args, **kwargs)
        else:
            ax.plot([], [])

    return fig, axes


def plot_2d(self, axes, *args, **kwargs):
    """Create an array of 2D plots.

    To avoid intefering with y-axis sharing, one-dimensional plots are
    created on a separate axis, which is monkey-patched onto the argument
    ax as the attribute ax.twin.

    Parameters
    ----------
    axes: plotting axes
        Can be:
            - list(str) if the x and y axes are the same
            - [list(str),list(str)] if the x and y axes are different
            - pandas.DataFrame(matplotlib.axes.Axes)
        If a pandas.DataFrame is provided as an existing set of axes, then
        this is used for creating the plot. Otherwise a new set of axes are
        created using the list or lists of strings.

    kind: dict, optional
        What kinds of plots to produce. Takes the keys 'diagonal'
        for the 1D plots and 'lower' and 'upper' for the 2D plots.
        The options for 'diagonal are:
            - 'kde'
            - 'hist'
            - 'astropyhist'
        The options for 'lower' and 'upper' are:
            - 'kde'
            - 'scatter'
            - 'hist'
            - 'fastkde'
        Default: {'diagonal': 'kde_1d',
                  'lower': 'kde_2d',
                  'upper':'scatter_2d'}

    diagonal_kwargs, lower_kwargs, upper_kwargs: dict, optional
        kwargs for the diagonal (1D)/lower or upper (2D) plots. This is
        useful when there is a conflict of kwargs for different kinds of
        plots.  Note that any kwargs directly passed to plot_2d will
        overwrite any kwarg with the same key passed to <sub>_kwargs.
        Default: {}

    Returns
    -------
    fig: matplotlib.figure.Figure
        New or original (if supplied) figure object

    axes: pandas.DataFrame of matplotlib.axes.Axes
        Pandas array of axes objects

    """
    default_kinds = {'diagonal': 'kde_1d',
                     'lower': 'kde_2d',
                     'upper': 'scatter_2d'}
    kind = kwargs.get('kind', default_kinds)
    local_kwargs = {pos: kwargs.pop('%s_kwargs' % pos, {})
                    for pos in default_kinds}
    kwargs['label'] = kwargs.get('label', self.label)

    self._set_automatic_limits()

    for pos in local_kwargs:
        local_kwargs[pos].update(kwargs)

    if not isinstance(axes, pandas.DataFrame):
        fig, axes = make_2d_axes(axes, tex=self.tex,
                                 upper=('upper' in kind),
                                 lower=('lower' in kind),
                                 diagonal=('diagonal' in kind))
    else:
        fig = axes.bfill().to_numpy().flatten()[0].figure

    for y, row in axes.iterrows():
        for x, ax in row.iteritems():
            if ax is not None:
                pos = ax.position
                lkwargs = local_kwargs.get(pos, {})
                lkwargs['kind'] = kind.get(pos, None)
                if lkwargs['kind'] is not None:
                    if x in self and y in self:
                        if x == y:
                            self[x].plot(ax=ax.twin, *args, **lkwargs)
                        else:
                            self.plot(x, y, ax=ax, *args, **lkwargs)
                else:
                    if x == y:
                        ax.twin.plot([], [])
                    else:
                        ax.plot([], [])

    return fig, axes
