from __future__ import annotations
from pandas.plotting import PlotAccessor as _PlotAccessor


def _process_docstring(doc):
    i = doc.find('\tax')
    extra = "\t- 'hist2d' : 2d distogram\n"\
            "\t- 'kde2d' : 2d Kernel Density Estimation plot\n"
    return doc[:i] + extra + doc[i:]


class PlotAccessor(_PlotAccessor):
    __doc__ = _process_docstring(_PlotAccessor.__doc__)
    _common_kinds = _PlotAccessor._common_kinds + ("hist1d", "kde1d")
    _series_kinds = _PlotAccessor._series_kinds + ()
    _dataframe_kinds = _PlotAccessor._dataframe_kinds + ("hist2d", "kde2d")
    _all_kinds = _common_kinds + _series_kinds + _dataframe_kinds

    def kde2d(self, x, y, **kwargs) -> PlotAccessor:
        return self(kind="kde2d", x=x, y=y, **kwargs)

    def hist2d(self, x, y, **kwargs) -> PlotAccessor:
        return self(kind="hist2d", x=x, y=y, **kwargs)

