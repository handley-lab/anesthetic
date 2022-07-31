from pandas.plotting._matplotlib.core import (ScatterPlot as _ScatterPlot,
                                              HexBinPlot as _HexBinPlot,
                                              MPLPlot)
from anesthetic.weighted_pandas import _WeightedObject


class _WeightedMPLPlot(MPLPlot):
    def __init__(self, data, *args, **kwargs):
        if isinstance(data, _WeightedObject):
            kwargs['weights'] = data.weights
        super().__init__(data, *args, **kwargs)


def _get_weights(kwds, data):
    if isinstance(data, _WeightedObject):
        kwds['weights'] = data.weights


class ScatterPlot(_ScatterPlot):
    # noqa: disable=D101
    def __init__(self, data, x, y, s=None, c=None, **kwargs) -> None:
        if isinstance(data, _WeightedObject):
            data = data.compress(kwargs.pop('ncompress', None))
            kwargs['alpha'] = kwargs.get('alpha', 0.5)
        super().__init__(data, x, y, s, c, **kwargs)


class HexBinPlot(_HexBinPlot):
    # noqa: disable=D101
    def __init__(self, data, x, y, C=None, **kwargs) -> None:
        if isinstance(data, _WeightedObject):
            C = '__weights'
            data[C] = data.weights
        super().__init__(data, x, y, C=C, **kwargs)
