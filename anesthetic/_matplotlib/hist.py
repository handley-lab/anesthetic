from pandas.plotting._matplotlib.hist import HistPlot as _HistPlot
from pandas.plotting._matplotlib.hist import KdePlot as _KdePlot
import pandas.plotting._matplotlib.hist
import numpy as np
from anesthetic.weighted_pandas import _WeightedObject
from pandas.core.dtypes.missing import (
    isna,
    remove_na_arraylike,
)
from pandas.plotting._matplotlib.core import MPLPlot


class HistPlot(_HistPlot):
    # noqa: disable=D101
    def _calculate_bins(self, data):
        nd_values = data._convert(datetime=True)._get_numeric_data()
        values = np.ravel(nd_values)
        values = values[~isna(values)]

        hist, bins = np.histogram(
            values, bins=self.bins, range=self.kwds.get("range", None),
            weights=self.kwds.get("weights", None)
        )
        return bins

    def _make_plot(self):
        if isinstance(self.data, _WeightedObject):
            self.kwds['weights'] = self.data.weights
        return super()._make_plot()


class KdePlot(HistPlot, _KdePlot):
    # noqa: disable=D101
    @classmethod
    def _plot(
        cls,
        ax,
        y,
        style=None,
        bw_method=None,
        ind=None,
        column_num=None,
        stacking_id=None,
        **kwds,
    ):
        from scipy.stats import gaussian_kde
        weights = kwds.pop("weights", None)

        y = remove_na_arraylike(y)
        gkde = gaussian_kde(y, bw_method=bw_method, weights=weights)

        y = gkde.evaluate(ind)
        lines = MPLPlot._plot(ax, ind, y, style=style, **kwds)
        return lines


def hist_frame(data, *args, **kwds):
    # noqa: disable=D103
    if isinstance(data, _WeightedObject):
        kwds['weights'] = data.weights
    return pandas.plotting._matplotlib.hist_frame(data, *args, **kwds)


def hist_series(data, *args, **kwds):
    # noqa: disable=D103
    if isinstance(data, _WeightedObject):
        kwds['weights'] = data.weights
    return pandas.plotting._matplotlib.hist_series(data, *args, **kwds)
