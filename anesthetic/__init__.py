"""Anesthetic: nested sampling post-processing."""
import anesthetic.samples
import anesthetic.plot
import anesthetic.read.chain
import anesthetic._version

__version__ = anesthetic._version.__version__
Samples = anesthetic.samples.Samples
MCMCSamples = anesthetic.samples.MCMCSamples
NestedSamples = anesthetic.samples.NestedSamples
make_2d_axes = anesthetic.plot.make_2d_axes
make_1d_axes = anesthetic.plot.make_1d_axes
read_chains = anesthetic.read.chain.read_chains
