from pandas.plotting._matplotlib import (  # noqa: F401
        PLOT_CLASSES,
        TYPE_CHECKING,
        __name__,
        plot,
        register,
        deregister
)


"""
from anesthetic._matplotlib.boxplot import (
    BoxPlot,
    boxplot,
    boxplot_frame,
    boxplot_frame_groupby,
)
"""
from anesthetic._matplotlib.hist import (
    HistPlot,
    #  KdePlot,
    hist_frame,
    hist_series,
)
"""
from anesthetic._matplotlib.misc import (
    andrews_curves,
    autocorrelation_plot,
    bootstrap_plot,
    lag_plot,
    parallel_coordinates,
    radviz,
    scatter_matrix,
)
from anesthetic._matplotlib.tools import table
"""

if TYPE_CHECKING:
    from pandas.plotting._matplotlib.core import MPLPlot  # noqa: F401

PLOT_CLASSES.pop('bar')
PLOT_CLASSES.pop('barh')
PLOT_CLASSES.pop('area')
PLOT_CLASSES.pop('pie')
PLOT_CLASSES['hist'] = HistPlot

__all__ = [
    "plot",
    "hist_series",
    "hist_frame",
    "boxplot",
    "boxplot_frame",
    "boxplot_frame_groupby",
    "table",
    "andrews_curves",
    "autocorrelation_plot",
    "bootstrap_plot",
    "lag_plot",
    "parallel_coordinates",
    "radviz",
    "scatter_matrix",
    "register",
    "deregister",
]
