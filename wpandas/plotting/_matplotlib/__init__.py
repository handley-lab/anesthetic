"""wpandas override of matplotlib backend for weighted samples."""
from pandas.plotting._matplotlib import (  # noqa: F401
        PLOT_CLASSES,
        __name__,
        __all__,
        plot,
        register,
        deregister
)
from wpandas.plotting._matplotlib.boxplot import (  # noqa: F401
    BoxPlot,
    boxplot,
    boxplot_frame,
)
from pandas.plotting._matplotlib import boxplot_frame_groupby  # noqa: F401
from wpandas.plotting._matplotlib.core import (  # noqa: F401
    AreaPlot,
    BarhPlot,
    BarPlot,
    HexBinPlot,
    LinePlot,
    PiePlot,
    ScatterPlot,
)
from wpandas.plotting._matplotlib.hist import (  # noqa: F401
    KdePlot,
    HistPlot,
    hist_frame,
    hist_series,
)
from wpandas.plotting._matplotlib.misc import (  # noqa: F401
    bootstrap_plot,
    scatter_matrix,
)
from pandas.plotting._matplotlib import (  # noqa: F401
    andrews_curves,
    autocorrelation_plot,
    lag_plot,
    parallel_coordinates,
    radviz,
    table,
)

PLOT_CLASSES["line"] = LinePlot
PLOT_CLASSES["bar"] = BarPlot
PLOT_CLASSES["barh"] = BarhPlot
PLOT_CLASSES["box"] = BoxPlot
PLOT_CLASSES["hist"] = HistPlot
PLOT_CLASSES["kde"] = KdePlot
PLOT_CLASSES["area"] = AreaPlot
PLOT_CLASSES["pie"] = PiePlot
PLOT_CLASSES["scatter"] = ScatterPlot
PLOT_CLASSES["hexbin"] = HexBinPlot
