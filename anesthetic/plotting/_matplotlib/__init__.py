"""anesthetic override of matplotlib backend for weighted samples."""
from wpandas.plotting._matplotlib import (  # noqa: F401
    PLOT_CLASSES,
    __name__,
    __all__,
    plot,
    register,
    deregister,
    bootstrap_plot,
    scatter_matrix,
    andrews_curves,
    autocorrelation_plot,
    lag_plot,
    parallel_coordinates,
    radviz,
    table,
    BoxPlot,
    boxplot,
    boxplot_frame,
    AreaPlot,
    BarhPlot,
    BarPlot,
    HexBinPlot,
    LinePlot,
    PiePlot,
    ScatterPlot,
    boxplot_frame_groupby,
    KdePlot,
    HistPlot,
    hist_frame,
    hist_series,
)
from anesthetic.plotting._matplotlib.core import (  # noqa: F401
    ScatterPlot2d,
)
from anesthetic.plotting._matplotlib.hist import (  # noqa: F401
    Kde1dPlot,
    FastKde1dPlot,
    Kde2dPlot,
    FastKde2dPlot,
    Hist1dPlot,
    Hist2dPlot,
)

PLOT_CLASSES['hist_1d'] = Hist1dPlot
PLOT_CLASSES['kde_1d'] = Kde1dPlot
PLOT_CLASSES['fastkde_1d'] = FastKde1dPlot
PLOT_CLASSES['hist_2d'] = Hist2dPlot
PLOT_CLASSES['kde_2d'] = Kde2dPlot
PLOT_CLASSES['fastkde_2d'] = FastKde2dPlot
PLOT_CLASSES['scatter_2d'] = ScatterPlot2d
