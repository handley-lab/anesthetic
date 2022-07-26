"""Anesthetic: nested sampling post-processing.

Key routines:

- ``MCMCSamples.build``
- ``MCMCSamples.read``
- ``NestedSamples.build``
- ``NestedSamples.read``

"""
import anesthetic.samples
import anesthetic.plot

from pandas import set_option as _set_option
from pandas.plotting._core import _backends
_set_option('plotting.backend', 'anesthetic._matplotlib')
_backends['matplotlib'] = _backends.pop('anesthetic._matplotlib')


MCMCSamples = anesthetic.samples.MCMCSamples
NestedSamples = anesthetic.samples.NestedSamples
make_2d_axes = anesthetic.plot.make_2d_axes
make_1d_axes = anesthetic.plot.make_1d_axes
