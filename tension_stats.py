#%%
from anesthetic.samples import Samples
from anesthetic.examples.perfect_ns import correlated_gaussian
from anesthetic import read_chains, make_2d_axes
import numpy as np
import matplotlib.pyplot as plt
#%%
# Creating mock datasets A, B and AB
nlive = 100
meanA = [0.1, 0.3, 0.5]
covA = [[.01, 0.009, 0], [0.009, .01, 0], [0, 0, 0.1]]
bounds = [[0, 1], [0,1], [0, 1]]
samplesA = correlated_gaussian(nlive, mean, cov, bounds) # output is Nested sampling run
#samplesA.gui()
#plt.show()

nlive = 100
meanB = [0.7, 0.2, 0.1]
covB = [[.01, 0.009, 0], [0.009, .01, 0], [0, 0, 0.1]]
#bounds = [[0, 1], [0,1], [0, 1]]
samplesB = correlated_gaussian(nlive, mean, cov, bounds)
# Make a plot
axes = samplesA.plot_2d([0,1,2])
samplesB.plot_2d(axes)


nsamples = 1000
Beta = 1
def tension_stats(A,B,AB,nsamples):
    # A, B and AB are datasets created with anesthetic.examples.perfect_ns_correlated_gaussian
    A.stats(nsamples)
    B.stats(nsamples)
    AB.stats(nsamples)
    # Create a new sample
    samples = Samples(index=A.index)
    samples['logR']=AB.logZ-A.logZ-B.logZ
    # do the same for logS, logI, d_G, p, return these values in a table
    return samples
samples = tension_stats(samplesA,samplesB)
# %%
