import pandas.plotting._matplotlib.hist
import numpy as np
import pandas as pd
from anesthetic.weighted_pandas import _WeightedObject


class HistPlot(pandas.plotting._matplotlib.hist.HistPlot):
    # noqa: disable=D101
    def _calculate_bins(self, data):
        nd_values = data._convert(datetime=True)._get_numeric_data()
        values = np.ravel(nd_values)
        values = values[~pd.isna(values)]

        hist, bins = np.histogram(
            values, bins=self.bins, range=self.kwds.get("range", None),
            weights=self.kwds.get("weights", None)
        )
        return bins

    def _make_plot(self):
        if isinstance(self.data, _WeightedObject):
            self.kwds['weights'] = self.data.weights
        return super()._make_plot()


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
