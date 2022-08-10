from pandas.plotting import PlotAccessor as _PlotAccessor


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
    # noqa: disable=D101
    __doc__ = _process_docstring(_PlotAccessor.__doc__)
    _common_kinds = _PlotAccessor._common_kinds \
        + ("hist_1d", "kde_1d", "fastkde_1d")
    _series_kinds = _PlotAccessor._series_kinds + ()
    _dataframe_kinds = _PlotAccessor._dataframe_kinds \
        + ("hist_2d", "kde_2d", "fastkde_2d", "scatter_2d")
    _all_kinds = _common_kinds + _series_kinds + _dataframe_kinds

    def hist_1d(self, **kwargs):
        """Histogram plot: See anesthetic.plot.hist_plot_1d."""
        return self(kind="hist_1d", **kwargs)

    def kde_1d(self, **kwargs):
        """KDE plot: See anesthetic.plot.kde_plot_1d."""
        return self(kind="kde_1d", **kwargs)

    def fastkde_1d(self, **kwargs):
        """KDE plot: See anesthetic.plot.fastkde_plot_1d."""
        return self(kind="fastkde_1d", **kwargs)

    def kde_2d(self, x, y, **kwargs):
        """KDE plot: See anesthetic.plot.kde_contour_plot_2d."""
        return self(kind="kde_2d", x=x, y=y, **kwargs)

    def fastkde_2d(self, x, y, **kwargs):
        """KDE plot: See anesthetic.plot.fastkde_contour_plot_2d."""
        return self(kind="fastkde_2d", x=x, y=y, **kwargs)

    def hist_2d(self, x, y, **kwargs):
        """Histogram plot: See anesthetic.plot.hist_plot_2d."""
        return self(kind="hist_2d", x=x, y=y, **kwargs)

    def scatter_2d(self, x, y, **kwargs):
        """Scatter plot: See anesthetic.plot.scatter_plot_2d."""
        return self(kind="scatter_2d", x=x, y=y, **kwargs)
