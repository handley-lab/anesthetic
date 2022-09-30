from pandas.plotting._matplotlib.hist import (HistPlot as _HistPlot,
                                              KdePlot as _KdePlot)
import pandas.plotting._matplotlib.hist
import numpy as np
from pandas.core.dtypes.missing import (
    isna,
    remove_na_arraylike,
)
from pandas.plotting._matplotlib.core import MPLPlot, PlanePlot
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from anesthetic.plotting._matplotlib.core import (
    _WeightedMPLPlot, _CompressedMPLPlot, _get_weights
)
from anesthetic.plot import (
    kde_contour_plot_2d,
    hist_plot_2d,
    fastkde_contour_plot_2d,
    kde_plot_1d,
    fastkde_plot_1d,
    hist_plot_1d,
)


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
    
    def _make_plot(self) -> None:
        from pandas.core.dtypes.common import is_extension_array_dtype
        from pandas.core.dtypes.missing import notna
        from pandas.plotting._matplotlib.groupby import (
            create_iter_data_given_by,
        )
        def remove_na_arraylike_idx(arr):
            if is_extension_array_dtype(arr):
                return notna(arr)
            return notna(np.asarray(arr))
        def reformat_hist_weights_given_y(y, weights, by):
            if y.shape != weights.shape:
                raise ValueError("data and weights must have the same shape")
            if by is not None and len(y.shape) > 1:
                return (
                    np.array([wcol.T[remove_na_arraylike_idx(ycol)] for ycol, wcol in zip(y.T, weights.T)]).T,
                )
            return weights[remove_na_arraylike_idx(y)]

        kwds = self.kwds.copy()
        weights = kwds.get("weights", None)
        data = (
            create_iter_data_given_by(self.data, self._kind)
            if self.by is not None
            else self.data
        )
        
        if weights is not None:
            print(data.shape)
            print(weights.shape)
            if np.ndim(weights) != 1:
                if data.shape != weights.T.shape:
                    raise ValueError("weights must be broadcastable to same shape as data.")
            else:
                weights = np.broadcast_to(weights, data.T.shape)
            # for i, (label, y) in enumerate(self._iter_data(data=data)):
            #     temp_weights = weights
            mask =  ~remove_na_arraylike_idx(data).T
            print(mask.shape)
            print(weights.shape)
            weights = np.ma.masked_array(weights, mask=mask)
            self.kwds["weights"] = np.squeeze(weights)
            print(weights)
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

    def _post_plot_logic(self, ax, data):
        ax.set_ylim(0,1)
        ax.set_yticks([])


class FastKde1dPlot(_CompressedMPLPlot, _HistPlot):
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
        return fastkde_plot_1d(ax, y, *args, **kwds)

    def _post_plot_logic(self, ax, data):
        ax.set_ylim(0,1)
        ax.set_yticks([])


class Hist1dPlot(HistPlot):
    # noqa: disable=D101
    @property
    def _kind(self) -> Literal["hist_1d"]:
        return "hist_1d"

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

        base = np.zeros(len(bins) - 1)
        bottom = bottom + cls._get_stacked_values(ax, stacking_id, base, kwds["label"])
        # ignore style 
        n, bins, patches = hist_plot_1d(ax, y, bins=bins, bottom=bottom, **kwds)
        cls._update_stacker(ax, stacking_id, n)
        return patches

    def _post_plot_logic(self, ax, data):
        ax.set_ylim(0,1)
        ax.set_xlim(self.bins[0], self.bins[-1])
        ax.set_yticks([])


class Kde2dPlot(_WeightedMPLPlot, PlanePlot):
    # noqa: disable=D101
    @property
    def _kind(self) -> Literal["kde_2d"]:
        return "kde_2d"

    def _make_plot(self):
        return kde_contour_plot_2d(
            self.axes[0],
            self.data[self.x].values,
            self.data[self.y].values,
            label=self.label,
            **self.kwds)


class FastKde2dPlot(_CompressedMPLPlot, PlanePlot):
    # noqa: disable=D101
    @property
    def _kind(self) -> Literal["fastkde_2d"]:
        return "fastkde_2d"

    def _make_plot(self):
        return fastkde_contour_plot_2d(
            self.axes[0],
            self.data[self.x].values,
            self.data[self.y].values,
            label=self.label,
            **self.kwds)


class Hist2dPlot(_WeightedMPLPlot, PlanePlot):
    # noqa: disable=D101
    @property
    def _kind(self) -> Literal["hist_2d"]:
        return "hist_2d"

    def _make_plot(self):
        return hist_plot_2d(
            self.axes[0],
            self.data[self.x].values,
            self.data[self.y].values,
            label=self.label,
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
