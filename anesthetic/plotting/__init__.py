"""Plotting public API.

.. autoclass:: anesthetic.plotting.PlotAccessor
"""
from anesthetic.plotting._core import PlotAccessor # noqa: 401
from anesthetic.plotting._matplotlib.core import ScatterPlot2d
from anesthetic.plotting._matplotlib.hist import (
    Kde1dPlot,
    FastKde1dPlot,
    Kde2dPlot,
    FastKde2dPlot,
    Hist1dPlot,
    Hist2dPlot,
)

import wpandas.plotting._matplotlib

wpandas.plotting._matplotlib.PLOT_CLASSES['hist_1d'] = Hist1dPlot
wpandas.plotting._matplotlib.PLOT_CLASSES['kde_1d'] = Kde1dPlot
wpandas.plotting._matplotlib.PLOT_CLASSES['fastkde_1d'] = FastKde1dPlot
wpandas.plotting._matplotlib.PLOT_CLASSES['hist_2d'] = Hist2dPlot
wpandas.plotting._matplotlib.PLOT_CLASSES['kde_2d'] = Kde2dPlot
wpandas.plotting._matplotlib.PLOT_CLASSES['fastkde_2d'] = FastKde2dPlot
wpandas.plotting._matplotlib.PLOT_CLASSES['scatter_2d'] = ScatterPlot2d
