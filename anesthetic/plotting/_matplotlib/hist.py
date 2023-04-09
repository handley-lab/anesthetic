import numpy as np
from wpandas.plotting._matplotlib.hist import KdePlot, HistPlot
from typing import Literal
from wpandas.plotting._matplotlib.core import (
    _WeightedMPLPlot, _CompressedMPLPlot
)
from anesthetic.plotting._matplotlib.core import _PlanePlot2d
from anesthetic.plot import (
    kde_contour_plot_2d,
    hist_plot_2d,
    fastkde_contour_plot_2d,
    kde_plot_1d,
    fastkde_plot_1d,
    hist_plot_1d,
)


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
