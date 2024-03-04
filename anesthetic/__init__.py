"""Anesthetic: nested sampling post-processing."""
import anesthetic.samples
import anesthetic.plot
import anesthetic.read.chain
import anesthetic.read.hdf

import pandas
import pandas.plotting._core
import pandas.plotting._misc
from anesthetic._format import _DataFrameFormatter
from anesthetic._version import __version__  # noqa: F401
# TODO: remove this when conda pandas version catches up
from packaging.version import parse
assert parse(pandas.__version__) >= parse('2.0.0')


def _anesthetic_override(_get_plot_backend):
    """Override the default backend.

    When _get_plot_backend asks for 'matplotlib' it will be directed to
    'anesthetic.plotting._matplotlib'. This is necessary since any users of
    :class:`anesthetic.weighted_pandas.WeightedSamples` should not be using the
    original backend.
    """
    def wrapper(backend=None):
        if backend == 'matplotlib':
            return _get_plot_backend('anesthetic.plotting._matplotlib')
        return _get_plot_backend(backend)
    return wrapper


# Override the two places where _get_plot_backend is defined
pandas.plotting._core._get_plot_backend = \
        _anesthetic_override(pandas.plotting._core._get_plot_backend)
pandas.plotting._misc._get_plot_backend = \
        _anesthetic_override(pandas.plotting._misc._get_plot_backend)

# Set anesthetic.plotting._matplotlib as the actual backend
pandas.options.plotting.backend = 'anesthetic.plotting._matplotlib'

pandas.io.formats.format.DataFrameFormatter = _DataFrameFormatter
pandas.options.display.max_colwidth = 14

Samples = anesthetic.samples.Samples
MCMCSamples = anesthetic.samples.MCMCSamples
NestedSamples = anesthetic.samples.NestedSamples
make_2d_axes = anesthetic.plot.make_2d_axes
make_1d_axes = anesthetic.plot.make_1d_axes

read_hdf = anesthetic.read.hdf.read_hdf
read_chains = anesthetic.read.chain.read_chains
