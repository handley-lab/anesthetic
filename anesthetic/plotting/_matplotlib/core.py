from pandas.plotting._matplotlib.core import (ScatterPlot as _ScatterPlot,
                                              HexBinPlot as _HexBinPlot,
                                              MPLPlot, PlanePlot)
from anesthetic.weighted_pandas import _WeightedObject
from anesthetic.plot import scatter_plot_2d
from typing import Literal
import numpy as np


def _get_weights(kwargs, data):
    if isinstance(data, _WeightedObject):
        kwargs['weights'] = data.weights


class _WeightedMPLPlot(MPLPlot):
    def __init__(self, data, *args, **kwargs):
        _get_weights(kwargs, data)
        super().__init__(data, *args, **kwargs)


def _compress_weights(kwargs, data):
    if isinstance(data, _WeightedObject):
        return data.compress(kwargs.pop('ncompress', None))
    else:
        return data


class _CompressedMPLPlot(MPLPlot):
    def __init__(self, data, *args, **kwargs):
        data = _compress_weights(kwargs, data)
        super().__init__(data, *args, **kwargs)


class ScatterPlot(_CompressedMPLPlot, _ScatterPlot):
    # noqa: disable=D101
    def __init__(self, data, x, y, s=None, c=None, **kwargs) -> None:
        if isinstance(data, _WeightedObject):
            kwargs['alpha'] = kwargs.get('alpha', 0.5)
        super().__init__(data, x, y, s, c, **kwargs)


class ScatterPlot2d(_CompressedMPLPlot, PlanePlot):
    # noqa: disable=D101
    @property
    def _kind(self) -> Literal["scatter_2d"]:
        return "scatter_2d"

    def _make_plot(self):
        return scatter_plot_2d(
            self.axes[0],
            self.data[self.x].values,
            self.data[self.y].values,
            label=self.label,
            **self.kwds)


class HexBinPlot(_HexBinPlot):
    # noqa: disable=D101
    def __init__(self, data, x, y, C=None, **kwargs) -> None:
        if isinstance(data, _WeightedObject):
            C = '__weights'
            data[C] = data.weights
            kwargs['reduce_C_function'] = np.sum
        super().__init__(data, x, y, C=C, **kwargs)
