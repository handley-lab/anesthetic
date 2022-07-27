"""Anesthetic: nested sampling post-processing.

Key routines:

- ``MCMCSamples.build``
- ``MCMCSamples.read``
- ``NestedSamples.build``
- ``NestedSamples.read``

"""
import anesthetic.samples
import anesthetic.plot

import pandas
import pandas.plotting._core
import pandas.plotting._misc


def _anesthetic_override(function):
    def wrapper(backend=None):
        if backend == 'matplotlib':
            return function('anesthetic._matplotlib')
        return function(backend)
    return wrapper


pandas.plotting._core._get_plot_backend = \
        _anesthetic_override(pandas.plotting._core._get_plot_backend)
pandas.plotting._misc._get_plot_backend = \
        _anesthetic_override(pandas.plotting._misc._get_plot_backend)

pandas.options.plotting.backend = 'anesthetic._matplotlib'


MCMCSamples = anesthetic.samples.MCMCSamples
NestedSamples = anesthetic.samples.NestedSamples
make_2d_axes = anesthetic.plot.make_2d_axes
make_1d_axes = anesthetic.plot.make_1d_axes
