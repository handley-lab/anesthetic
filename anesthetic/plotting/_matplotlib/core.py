from typing import Literal
import numpy as np
from matplotlib import pyplot as plt
from pandas.core.dtypes.generic import ABCMultiIndex
import pandas.core.common as com
from pandas.plotting._matplotlib.core import (ScatterPlot as _ScatterPlot,
                                              HexBinPlot as _HexBinPlot,
                                              LinePlot as _LinePlot,
                                              BarPlot as _BarPlot,
                                              BarhPlot as _BarhPlot,
                                              AreaPlot as _AreaPlot,
                                              PiePlot as _PiePlot,
                                              MPLPlot, PlanePlot)
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by
from pandas.io.formats.printing import pprint_thing
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
            if isinstance(self.data.drop_weights().index, ABCMultiIndex):
                name = self.data.drop_weights().index.names
                if com.any_not_none(*name):
                    name = ",".join([pprint_thing(x) for x in name])
                else:
                    name = None
            else:
                name = self.data.drop_weights().index.name
                if name is not None:
                    name = pprint_thing(name)

            # GH 45145, override the default axis label if one is provided.
            index_name = self._get_custom_index_name()
            if index_name is not None:
                name = pprint_thing(index_name)

            return name
        else:
            return super()._get_index_name()

    def _get_xticks(self, convert_period: bool = False):
        if isinstance(self.data, _WeightedObject):
            return self.data.drop_weights().index._mpl_repr()
        else:
            return super()._get_xticks(convert_period)


def _compress_weights(kwargs, data):
    if isinstance(data, _WeightedObject):
        return data.compress(kwargs.pop('ncompress', 'equal'))
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


class _PlanePlot2d(PlanePlot):

    def _make_plot(self):
        if self.colormap is not None:
            self.kwds['cmap'] = plt.get_cmap(self.colormap)
        colors = self._get_colors()
        data = (
            create_iter_data_given_by(self.data, self.kind)  # safe for now
            if self.by is not None
            else self.data
        )
        x = data[self.x]
        y = data[self.y]
        if x.name:  # transfer the tex labels
            self.x = x.name
        if y.name:
            self.y = y.name
        ax = self._get_ax(0)  # another one of these hard-coded 0s

        kwds = self.kwds.copy()
        label = pprint_thing(self.label)
        kwds["label"] = label

        style, kwds = self._apply_style_colors(colors, kwds, 0, label)
        if style is not None:
            raise TypeError("'style' keyword argument is not "
                            f"supported by {self._kind}")
        self._plot(ax, x.values, y.values, **kwds)

    def _args_adjust(self):
        pass


class ScatterPlot2d(_CompressedMPLPlot, _PlanePlot2d):
    # noqa: disable=D101
    @property
    def _kind(self) -> Literal["scatter_2d"]:
        return "scatter_2d"

    @classmethod
    def _plot(cls, ax, x, y, **kwds):
        return scatter_plot_2d(ax, x, y, **kwds)


class HexBinPlot(_HexBinPlot):
    # noqa: disable=D101
    def __init__(self, data, x, y, C=None, **kwargs) -> None:
        if isinstance(data, _WeightedObject):
            C = '__weights'
            data[C] = data.get_weights()
            kwargs['reduce_C_function'] = np.sum
            if 'mincnt' not in kwargs:
                kwargs['mincnt'] = np.finfo(np.float64).tiny
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
