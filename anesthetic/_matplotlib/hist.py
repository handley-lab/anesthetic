from pandas.plotting._matplotlib.hist import (HistPlot as _HistPlot,
                                              KdePlot as _KdePlot)
import pandas.plotting._matplotlib.hist
import numpy as np
from pandas.core.dtypes.missing import (
    isna,
    remove_na_arraylike,
)
from pandas.plotting._matplotlib.core import MPLPlot, PlanePlot
from typing import Literal
from anesthetic._matplotlib.core import _WeightedMPLPlot, _get_weights
from anesthetic.plot import kde_contour_plot_2d, hist_plot_2d


class HistPlot(_WeightedMPLPlot, _HistPlot):
    # noqa: disable=D101
    def _calculate_bins(self, data):
        nd_values = data._convert(datetime=True)._get_numeric_data()
        values = np.ravel(nd_values)
        weights = self.kwds.get("weights", None)
        if weights is not None:
            try:
                weights = np.broadcast_to(weights[:, None], nd_values.shape)
            except ValueError:
                print(nd_values.shape, weights.shape)
                pass
            weights = np.ravel(weights)
            weights = weights[~isna(values)]

        values = values[~isna(values)]

        hist, bins = np.histogram(
            values, bins=self.bins, range=self.kwds.get("range", None),
            weights=weights
        )
        return bins


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


class Kde2dPlot(_WeightedMPLPlot, PlanePlot):
    @property
    def _kind(self) -> Literal["kde_2d"]:
        return "kde_2d"

    def _make_plot(self):
        return kde_contour_plot_2d(
            self.axes[0],
            self.data[self.x].values,
            self.data[self.y].values,
            **self.kwds)


class Hist2dPlot(_WeightedMPLPlot, PlanePlot):
    @property
    def _kind(self) -> Literal["hist_2d"]:
        return "hist_2d"

    def _make_plot(self):
        return hist_plot_2d(
            self.axes[0],
            self.data[self.x].values,
            self.data[self.y].values,
            **self.kwds,
        )


def hist_frame(data, *args, **kwds):
    # noqa: disable=D103
    _get_weights(kwds, data)
    return pandas.plotting._matplotlib.hist_frame(data, *args, **kwds)


def hist_series(data, *args, **kwds):
    # noqa: disable=D103
    _get_weights(kwds, data)
    return pandas.plotting._matplotlib.hist_series(data, *args, **kwds)
