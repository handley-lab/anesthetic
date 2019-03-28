import numpy
from numpy import pi
from numpy.linalg import inv, slogdet
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

nDims = 2
nDerived = 0
mu = numpy.array([1, 1])
cov = 0.1**2*numpy.array([[1,-0.9],[-0.9,1]])

def likelihood(theta):
    """ Simple Gaussian Likelihood"""
    logL = -slogdet(2*pi*cov)[0]/2 - (theta-mu) @ inv(cov) @ (theta-mu) / 2
    return logL, []

def prior(hypercube):
    """ Uniform prior from [-1,1]^D. """
    return UniformPrior(-2, 2)(hypercube)


settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'example'
settings.nlive = 500
settings.read_resume = False

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
output.make_paramnames_files([('A', 'A'), ('B', 'B')])
