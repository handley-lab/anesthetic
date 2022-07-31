"""anesthetic override of matplotlib backend for weighted samples."""
from pandas.plotting._matplotlib import (  # noqa: F401
        PLOT_CLASSES,
        __name__,
        __all__,
        plot,
        register,
        deregister
)
from anesthetic._matplotlib.boxplot import (  # noqa: F401
    BoxPlot,
    boxplot,
    boxplot_frame,
)
from pandas.plotting._matplotlib import boxplot_frame_groupby  # noqa: F401
from anesthetic._matplotlib.core import (  # noqa: F401
    ScatterPlot,
    HexBinPlot,
)
from anesthetic._matplotlib.hist import (  # noqa: F401
    HistPlot,
    KdePlot,
    Kde2dPlot,
    Hist2dPlot,
    hist_frame,
    hist_series,
)
from anesthetic._matplotlib.misc import (  # noqa: F401
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

PLOT_CLASSES['hist'] = HistPlot
PLOT_CLASSES['kde'] = KdePlot
PLOT_CLASSES['box'] = BoxPlot
PLOT_CLASSES['scatter'] = ScatterPlot
PLOT_CLASSES['hexbin'] = HexBinPlot

PLOT_CLASSES['kde2d'] = Kde2dPlot
PLOT_CLASSES['hist2d'] = Hist2dPlot
