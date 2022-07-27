from pandas.plotting._matplotlib import (  # noqa: F401
        PLOT_CLASSES,
        TYPE_CHECKING,
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
"""
from pandas.plotting._matplotlib.core import (
    AreaPlot,
    BarhPlot,
    BarPlot,
    HexBinPlot,
    LinePlot,
    PiePlot,
    ScatterPlot,
)
"""
from anesthetic._matplotlib.hist import (  # noqa: F401
    HistPlot,
    KdePlot,
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

if TYPE_CHECKING:
    from pandas.plotting._matplotlib.core import MPLPlot  # noqa: F401

PLOT_CLASSES['hist'] = HistPlot
PLOT_CLASSES['kde'] = KdePlot
PLOT_CLASSES['box'] = BoxPlot
