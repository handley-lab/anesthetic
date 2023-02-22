from typing import Literal
import numpy as np
from pandas.plotting._matplotlib.core import (ScatterPlot as _ScatterPlot,
                                              HexBinPlot as _HexBinPlot,
                                              LinePlot as _LinePlot,
                                              BarPlot as _BarPlot,
                                              BarhPlot as _BarhPlot,
                                              AreaPlot as _AreaPlot,
                                              PiePlot as _PiePlot,
                                              MPLPlot, PlanePlot)
from anesthetic.weighted_pandas import _WeightedObject
from anesthetic.plot import scatter_plot_2d


def _get_weights(kwargs, data):
    if isinstance(data, _WeightedObject):
        kwargs['weights'] = data.get_weights()


class _WeightedMPLPlot(MPLPlot):

    _default_rot = None

    def __init__(self, data, *args, **kwargs):
        _get_weights(kwargs, data)
        super().__init__(data, *args, **kwargs)

    def _get_index_name(self):
        if isinstance(self.data, _WeightedObject):
            return np.squeeze(self.data.drop_weights().index.names)
        else:
            return super()._get_index_name()

    def _get_xticks(self, convert_period: bool = False):
        if isinstance(self.data, _WeightedObject):
            return self.data.drop_weights().index._mpl_repr()
        else:
            return super()._get_xticks(convert_period)


def _compress_weights(kwargs, data):
    if isinstance(data, _WeightedObject):
        return data.compress(kwargs.pop('ncompress', True))
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
        if self.colormap is not None:
            self.kwds['cmap'] = self.colormap
        return scatter_plot_2d(
            self.axes[0],
            self.data[self.x].values,
            self.data[self.y].values,
            label=self.label,
            **self.kwds)

    def _validate_color_args(self):
        pass


class HexBinPlot(_HexBinPlot):
    # noqa: disable=D101
    def __init__(self, data, x, y, C=None, **kwargs) -> None:
        if isinstance(data, _WeightedObject):
            C = '__weights'
            data[C] = data.get_weights()
            kwargs['reduce_C_function'] = np.sum
        super().__init__(data, x, y, C=C, **kwargs)


class LinePlot(_LinePlot, _WeightedMPLPlot):
    # noqa: disable=D101
    pass


class PiePlot(_PiePlot):
    # noqa: disable=D101
    def __init__(self, data, kind=None, **kwargs) -> None:
        if isinstance(data, _WeightedObject):
            labels = data.drop_weights().index._mpl_repr()
            kwargs['labels'] = kwargs.get('labels', labels)
        super().__init__(data, kind=kind, **kwargs)


class BarPlot(_BarPlot, _WeightedMPLPlot):
    # noqa: disable=D101
    def _decorate_ticks(self, ax, name, ticklabels, start_edge, end_edge):
        super()._decorate_ticks(ax, name, ticklabels, start_edge, end_edge)
        if isinstance(self.data, _WeightedObject):
            if self.xticks is None:
                ax.set_xticklabels(self._get_xticks())


class BarhPlot(_BarhPlot, _WeightedMPLPlot):
    # noqa: disable=D101
    def _decorate_ticks(self, ax, name, ticklabels, start_edge, end_edge):
        super()._decorate_ticks(ax, name, ticklabels, start_edge, end_edge)
        if isinstance(self.data, _WeightedObject):
            if self.yticks is None:
                ax.set_yticklabels(self._get_xticks())


class AreaPlot(_AreaPlot, _WeightedMPLPlot):
    # noqa: disable=D101
    pass
