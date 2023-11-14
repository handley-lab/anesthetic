from __future__ import annotations
from matplotlib import pyplot as plt
import numpy as np
from pandas.plotting._matplotlib.hist import (HistPlot as _HistPlot,
                                              KdePlot as _KdePlot)
import pandas.plotting._matplotlib.hist
from pandas.core.dtypes.missing import (
    isna,
    remove_na_arraylike,
)
from pandas.plotting._matplotlib.core import MPLPlot
from typing import Literal
from anesthetic.plotting._matplotlib.core import (
    _WeightedMPLPlot, _CompressedMPLPlot, _PlanePlot2d, _get_weights
)
from anesthetic.plot import (
    kde_contour_plot_2d,
    hist_plot_2d,
    fastkde_contour_plot_2d,
    kde_plot_1d,
    fastkde_plot_1d,
    hist_plot_1d,
    quantile_plot_interval,
)
from anesthetic.utils import quantile, histogram_bin_edges


class HistPlot(_WeightedMPLPlot, _HistPlot):

    # noqa: disable=D101
    def _args_adjust(self) -> None:
        if (
                hasattr(self, 'bins') and
                isinstance(self.bins, str) and
                self.bins in ['fd', 'scott', 'sqrt']
        ):
            self.bins = self._calculate_bins(self.data)
        super()._args_adjust()

    # noqa: disable=D101
    def _calculate_bins(self, data):
        if self.logx:
            data = np.log10(data)
            if 'range' in self.kwds and self.kwds['range'] is not None:
                xmin, xmax = self.kwds['range']
                self.kwds['range'] = (np.log10(xmin), np.log10(xmax))
        nd_values = data.infer_objects(copy=False)._get_numeric_data()
        values = np.ravel(nd_values)
        weights = self.kwds.get("weights", None)
        if weights is not None:
            try:
                weights = np.broadcast_to(weights[:, None], nd_values.shape)
            except ValueError:
                pass
            weights = np.ravel(weights)
            weights = weights[~isna(values)]

        values = values[~isna(values)]

        if isinstance(self.bins, str) and self.bins in ['fd', 'scott', 'sqrt']:
            bins = histogram_bin_edges(
                values,
                weights=weights,
                bins=self.bins,
                beta=self.kwds.pop('beta', 'equal'),
                range=self.kwds.get('range', None)
            )
        else:
            bins = np.histogram_bin_edges(
                values,
                weights=weights,
                bins=self.bins,
                range=self.kwds.get('range', None)
            )
        if self.logx:
            bins = 10**bins
            if 'range' in self.kwds and self.kwds['range'] is not None:
                self.kwds['range'] = (xmin, xmax)
        return bins

    def _get_colors(self, num_colors=None, color_kwds='color'):
        if (self.colormap is not None and self.kwds.get(color_kwds) is None
           and (num_colors is None or num_colors == 1)):
            return [plt.get_cmap(self.colormap)(0.68)]
        return super()._get_colors(num_colors, color_kwds)

    def _post_plot_logic(self, ax, data):
        ax.set_xlabel(self.xlabel)
        ax.set_yticks([])
        ax.set_ylim(bottom=0)


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

    def _post_plot_logic(self, ax, data):
        ax.set_yticks([])
        ax.set_ylim(bottom=0)


class Kde1dPlot(KdePlot):
    # noqa: disable=D101
    @property
    def _kind(self) -> Literal["kde_1d"]:
        return "kde_1d"

    # noqa: disable=D101
    @classmethod
    def _plot(
        cls,
        ax,
        y,
        style=None,
        ind=None,
        column_num=None,
        stacking_id=None,
        **kwds,
    ):
        args = (style,) if style is not None else tuple()
        return kde_plot_1d(ax, y, *args, **kwds)


class FastKde1dPlot(_CompressedMPLPlot, Kde1dPlot):
    # noqa: disable=D101
    @property
    def _kind(self) -> Literal["fastkde_1d"]:
        return "fastkde_1d"

    @classmethod
    def _plot(
        cls,
        ax,
        y,
        style=None,
        ind=None,
        column_num=None,
        stacking_id=None,
        **kwds,
    ):
        args = (style,) if style is not None else tuple()
        # weights and bw_method are not valid for fastkde
        kwds.pop('weights', None)
        kwds.pop('bw_method', None)
        return fastkde_plot_1d(ax, y, *args, **kwds)


class Hist1dPlot(HistPlot):
    # noqa: disable=D101
    @property
    def _kind(self) -> Literal["hist_1d"]:
        return "hist_1d"

    def __init__(
            self,
            data,
            bins: str | int | np.ndarray | list[np.ndarray] = 'fd',
            bottom: int | np.ndarray = 0,
            **kwargs,
    ) -> None:
        super().__init__(data, bins=bins, bottom=bottom, **kwargs)

    def _calculate_bins(self, data):
        if 'range' not in self.kwds or self.kwds['range'] is None:
            q = self.kwds.get('q', 5)
            q = quantile_plot_interval(q=q)
            weights = self.kwds.get('weights', None)
            xmin = quantile(data, q[0], weights)
            xmax = quantile(data, q[-1], weights)
            self.kwds['range'] = (xmin, xmax)
            bins = super()._calculate_bins(data)
            self.kwds.pop('range')
        else:
            bins = super()._calculate_bins(data)
        return bins

    @classmethod
    def _plot(
        cls,
        ax,
        y,
        style=None,
        bottom=0,
        column_num=None,
        stacking_id=None,
        *,
        bins,
        **kwds,
    ):
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(bins) - 1)

        if not isinstance(bins, str):
            base = np.zeros(len(bins) - 1)
            bottom = bottom + cls._get_stacked_values(ax, stacking_id,
                                                      base, kwds["label"])

        # ignore style
        n, bins, patches = hist_plot_1d(ax, y, bins=bins,
                                        bottom=bottom, **kwds)
        cls._update_stacker(ax, stacking_id, n)
        return patches


class Kde2dPlot(_WeightedMPLPlot, _PlanePlot2d):
    # noqa: disable=D101
    @property
    def _kind(self) -> Literal["kde_2d"]:
        return "kde_2d"

    @classmethod
    def _plot(cls, ax, x, y, **kwds):
        return kde_contour_plot_2d(ax, x, y, **kwds)


class FastKde2dPlot(_CompressedMPLPlot, _PlanePlot2d):
    # noqa: disable=D101
    @property
    def _kind(self) -> Literal["fastkde_2d"]:
        return "fastkde_2d"

    @classmethod
    def _plot(cls, ax, x, y, **kwds):
        return fastkde_contour_plot_2d(ax, x, y, **kwds)


class Hist2dPlot(_WeightedMPLPlot, _PlanePlot2d):
    # noqa: disable=D101
    @property
    def _kind(self) -> Literal["hist_2d"]:
        return "hist_2d"

    @classmethod
    def _plot(cls, ax, x, y, **kwds):
        return hist_plot_2d(ax, x, y, **kwds)


def hist_frame(data, *args, **kwds):
    # noqa: disable=D103
    _get_weights(kwds, data)
    return pandas.plotting._matplotlib.hist_frame(data, *args, **kwds)


def hist_series(data, *args, **kwds):
    # noqa: disable=D103
    _get_weights(kwds, data)
    return pandas.plotting._matplotlib.hist_series(data, *args, **kwds)
