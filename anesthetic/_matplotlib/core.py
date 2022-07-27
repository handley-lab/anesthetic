from pandas.plotting._matplotlib.core import ScatterPlot as _ScatterPlot
from pandas.plotting._matplotlib.core import HexBinPlot as _HexBinPlot
from anesthetic.weighted_pandas import _WeightedObject


class ScatterPlot(_ScatterPlot):
    # noqa: disable=D101
    def __init__(self, data, x, y, s=None, c=None, **kwargs) -> None:
        if isinstance(data, _WeightedObject):
            data = data.compress()
            kwargs['alpha'] = kwargs.get('alpha', 0.5)
        super().__init__(data, x, y, s, c, **kwargs)


class HexBinPlot(_HexBinPlot):
    # noqa: disable=D101
    def __init__(self, data, x, y, C=None, **kwargs) -> None:
        if isinstance(data, _WeightedObject):
            C = '__weights'
            data[C] = data.weights
        super().__init__(data, x, y, C=C, **kwargs)
