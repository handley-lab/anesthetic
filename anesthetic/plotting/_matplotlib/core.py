from typing import Literal
from matplotlib import pyplot as plt
from pandas.plotting._matplotlib.core import PlanePlot
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by
from pandas.io.formats.printing import pprint_thing
from anesthetic.plot import scatter_plot_2d
from wpandas.plotting._matplotlib.core import _CompressedMPLPlot


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
        x = data[self.x].values
        y = data[self.y].values
        ax = self._get_ax(0)  # another one of these hard-coded 0s

        kwds = self.kwds.copy()
        label = pprint_thing(self.label)
        kwds["label"] = label

        style, kwds = self._apply_style_colors(colors, kwds, 0, label)
        if style is not None:
            raise TypeError("'style' keyword argument is not "
                            f"supported by {self._kind}")
        self._plot(ax, x, y, **kwds)


class ScatterPlot2d(_CompressedMPLPlot, _PlanePlot2d):
    # noqa: disable=D101
    @property
    def _kind(self) -> Literal["scatter_2d"]:
        return "scatter_2d"

    @classmethod
    def _plot(cls, ax, x, y, **kwds):
        return scatter_plot_2d(ax, x, y, **kwds)
