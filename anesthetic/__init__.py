"""Anesthetic: nested sampling post-processing.

Key routines:

- ``MCMCSamples.build``
- ``MCMCSamples.read``
- ``NestedSamples.build``
- ``NestedSamples.read``

"""
import anesthetic.samples
import anesthetic.plot


MCMCSamples = anesthetic.samples.MCMCSamples
NestedSamples = anesthetic.samples.NestedSamples
make_2d_axes = anesthetic.plot.make_2d_axes
make_1d_axes = anesthetic.plot.make_1d_axes
get_legend_proxy = anesthetic.plot.get_legend_proxy
