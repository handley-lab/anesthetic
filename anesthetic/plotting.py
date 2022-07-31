from __future__ import annotations
from pandas.plotting import PlotAccessor as _PlotAccessor


def _process_docstring(doc):
    i = doc.find('    ax')
    e = "    - 'hist_2d' : 2d histogram (DataFrame only)\n"\
        "    - 'kde_2d' : 2d Kernel Density Estimation plot (DataFrame only)\n"\
        "    - 'fastkde_2d' : 2d Kernel Density Estimation plot with fastkde package (DataFrame only)\n"
    return doc[:i] + e + doc[i:]


class PlotAccessor(_PlotAccessor):
    __doc__ = _process_docstring(_PlotAccessor.__doc__)
    _common_kinds = _PlotAccessor._common_kinds + ("hist1d", "kde1d")
    _series_kinds = _PlotAccessor._series_kinds + ()
    _dataframe_kinds = _PlotAccessor._dataframe_kinds + ("hist_2d", "kde_2d", "fastkde_2d")
    _all_kinds = _common_kinds + _series_kinds + _dataframe_kinds

    def kde_2d(self, x, y, **kwargs) -> PlotAccessor:
        return self(kind="kde_2d", x=x, y=y, **kwargs)

    def fastkde_2d(self, x, y, **kwargs) -> PlotAccessor:
        return self(kind="fastkde_2d", x=x, y=y, **kwargs)

    def hist_2d(self, x, y, **kwargs) -> PlotAccessor:
        return self(kind="hist_2d", x=x, y=y, **kwargs)
