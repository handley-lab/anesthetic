import pandas
import pandas.plotting._core
import pandas.plotting._misc
import wpandas.core.series
import wpandas.core.frame


def _wpandas_override(_get_plot_backend):
    """Override the default backend.

    When _get_plot_backend asks for 'matplotlib' it will be directed to
    'wpandas.plotting._matplotlib'. This is necessary since any users of
    :class:`wpandas.WeightedSamples` should not be using the original backend.
    """
    def wrapper(backend=None):
        if backend == 'matplotlib':
            return _get_plot_backend('wpandas.plotting._matplotlib')
        return _get_plot_backend(backend)
    return wrapper


# Override the two places where _get_plot_backend is defined
pandas.plotting._core._get_plot_backend = \
        _wpandas_override(pandas.plotting._core._get_plot_backend)
pandas.plotting._misc._get_plot_backend = \
        _wpandas_override(pandas.plotting._misc._get_plot_backend)

# Set wpandas.plotting._matplotlib as the actual backend
pandas.options.plotting.backend = 'wpandas.plotting._matplotlib'

WeightedSeries = wpandas.core.series.WeightedSeries
WeightedDataFrame = wpandas.core.frame.WeightedDataFrame
