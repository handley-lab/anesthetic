#%%
from anesthetic.examples.perfect_ns import correlated_gaussian
from anesthetic import read_chains, make_2d_axes
import numpy as np
import matplotlib.pyplot as plt
#%%
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

#%%
statsA = samplesA.stats(nsamples=1000)
statsB = samplesB.stats(nsamples=1000)
params = ['logZ', 'D_KL', 'logL_P', 'd_G']
fig, axes = make_2d_axes(params, figsize=(6, 6), facecolor='w', upper=False)
statsA.plot_2d(axes, label="Dataset A")
statsB.plot_2d(axes, label="Dataset B")
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes), len(axes)), loc='upper right')

# %%
def testGaussian(theta,mu,logLmax,v,cov):
    """
    Inputs:
    theta: parameters
    mu: centre of Gaussian likelihood
    logLmax: log of Gaussian peak
    V: prior volume, for hypercube V=1
    cov: covariance of gaussian in parameters
    """
    gaussLogL = logLmax - 1/2*(theta-mu)*np.linalg.inv(cov)*(theta-mu)
    gaussLogP = -1/2*np.log(abs(2*np.pi*cov))-1/2*(theta-mu)*np.linalg.inv(cov)*(theta-mu)
    gaussLogZ = logLmax+1/2*np.log(abs(2*np.pi*cov))-np.log(v)
    D=np.log(v)-1/2*(len(theta)+np.log(abs(2*np.pi*cov)))
    return gaussLogL,gaussLogP,gaussLogZ,D

testGaussian(theta=np.array([0.111955,0.312646	,0.542911]),mu=meanA,logLmax=samplesA['logL'].max(),v=1,cov=np.array(covA))

# %%
