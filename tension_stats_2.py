#%%
from anesthetic.samples import Samples
from anesthetic.examples.perfect_ns import correlated_gaussian
from anesthetic import read_chains, make_2d_axes
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
#%%
# Creating mock datasets A, B and AB
# Creating mock dataset A
nlive = 100
meanA = [0.1, 0.3, 0.5]
covA = np.asmatrix([[.01, 0.009, 0], [0.009, .01, 0], [0, 0, 0.1]])
bounds = [[0, 1], [0,1], [0, 1]]
samplesA = correlated_gaussian(nlive, meanA, covA, bounds) # output is Nested sampling run
#samplesA.gui()
#plt.show()

# Creating mock dataset B
nlive = 100
meanB = [0.7, 0.2, 0.1]
covB = np.asmatrix([[.01, 0.009, 0], [0.009, .01, 0], [0, 0, 0.1]])
#bounds = [[0, 1], [0,1], [0, 1]]
samplesB = correlated_gaussian(nlive, meanB, covB, bounds)
# Make a plot
axes = samplesA.plot_2d([0,1,2])
samplesB.plot_2d(axes)

# Calculate covariance of AB using equation 19 from paper
covA_inv = linalg.inv(covA)
covB_inv = linalg.inv(covB)
covAB_inv = covA_inv+covB_inv
covAB = linalg.inv(covAB_inv)

# Calculate mean of AB using equation 20 from paper
meanAB = covAB*(covA_inv*(np.asmatrix(meanA).transpose())+covB_inv*(np.asmatrix(meanB).transpose()))
meanAB = np.asarray(meanAB.transpose())

# Creating mock dataset AB
nlive = 100
# Matching the input for the func correlated_gaussian
meanAB_=meanAB.flatten().tolist()
covAB_=covAB.tolist()

samplesAB = correlated_gaussian(nlive, meanAB_, covAB_,bounds)
samplesAB.plot_2d(axes)
#%%
# Set parameters
nsamples = 1000
beta = 1

def tension_stats(A,B,AB,nsamples):
    # A, B and AB are datasets created with anesthetic.examples.perfect_ns_correlated_gaussian
    statsA= A.stats(nsamples,beta)
    statsB=B.stats(nsamples,beta)
    statsAB=AB.stats(nsamples,beta)
    # Create a new sample
    samples = Samples(index=statsA.index)
    logR = statsAB.logZ-statsA.logZ-statsB.logZ
    samples['logR']=logR
    logI  = statsA.D_KL + statsB.D_KL - statsAB.D_KL
    samples['logI']=logI
    samples['logS']=logR-logI
    samples['d_G']=statsAB.d_G
    samples['logL_P']=statsAB.logL_P
    # do the same for logS, logI, d_G, p, return these values in a table
    return samples,statsA

samples,statsA = tension_stats(samplesA,samplesB,samplesAB,nsamples)
print(samples)
# %%
# Calculate the exact solution of logR
def get_logR(meanA,meanB,covA,covB,V):
    #return -1/2*np.asmatrix(meanA-meanB)*linalg.inv((covA+covB))*(meanA-meanB)-1/2*np.log(abs(2*np.pi*(covA+covB)))+np.log(V)
    return -1/2*np.asmatrix(meanA_-meanB_)*linalg.inv((covA+covB))*(np.asmatrix((meanA_-meanB_)).transpose())-1/2*np.log(linalg.det(2*np.pi*(covA+covB)))+np.log(V)


V=1
meanA_ = np.array(meanA)
meanB_ = np.array(meanB)
logR_exact = get_logR(meanA_,meanB_,covA,covB,V)
print(logR_exact)
# %%
def get_logS(meanA,meanB,covA,covB,d):
    return d/2 - 1/2*np.asmatrix(meanA-meanB)*linalg.inv((covA+covB))*(np.asmatrix(meanA-meanB).transpose())
logS_exact = get_logS(meanA_,meanB_,covA,covB,3)
print(logS_exact)
# %%
def get_logI(covA,covB,d,V):
    return -d/2 -1/2* - 1/2*np.log(linalg.det(2*np.pi*(covA+covB))) + np.log(V)
logI_exact = get_logI(covA,covB,3,1)
print(logI_exact)
# %%
