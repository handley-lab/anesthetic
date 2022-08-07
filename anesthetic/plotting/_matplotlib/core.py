from pandas.plotting._matplotlib.core import (ScatterPlot as _ScatterPlot,
                                              HexBinPlot as _HexBinPlot,
                                              MPLPlot)
from anesthetic.weighted_pandas import _WeightedObject
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


class HexBinPlot(_HexBinPlot):
    # noqa: disable=D101
    def __init__(self, data, x, y, C=None, **kwargs) -> None:
        if isinstance(data, _WeightedObject):
            C = '__weights'
            data[C] = data.weights
            kwargs['reduce_C_function'] = np.sum
        super().__init__(data, x, y, C=C, **kwargs)
