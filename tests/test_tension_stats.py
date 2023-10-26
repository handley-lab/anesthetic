
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
nlive = 1000
meanA = [0.1, 0.3, 0.5]
covA = np.array([[.01, 0.009, 0], [0.009, .01, 0], [0, 0, 0.1]])*0.01
bounds = [[0, 1], [0,1], [0, 1]]
logLmaxA = 0
samplesA = correlated_gaussian(nlive, meanA, covA, bounds, logLmaxA) # output is Nested sampling run
#samplesA.gui()
#plt.show()

# Creating mock dataset B

# Datasets being incompatible 
meanB = [0.15, 0.25, 0.45]
# Datasets being compatible
meanB = [0.1, 0.3, 0.5]

covB = np.array([[.01, -0.009, 0.01], [-0.009, .01, -0.001], [0.01, -0.001, 0.1]])*0.01
logLmaxB = 0
samplesB = correlated_gaussian(nlive, meanB, covB, bounds,logLmaxB)
# Make a plot
axes = samplesA.plot_2d([0,1,2])
samplesB.plot_2d(axes)

# Calculate covariance of AB using equation 19 from paper
covA_inv = linalg.inv(covA)
covB_inv = linalg.inv(covB)
covAB_inv = covA_inv+covB_inv
covAB = linalg.inv(covAB_inv)

# Calculate mean of AB using equation 20 from paper
#meanAB1 = covAB@(covA_inv@meanA+covB_inv@meanB)
meanAB = covAB@(np.linalg.solve(covA,meanA)+np.linalg.solve(covB,meanB))
#meanAB = np.asarray(meanAB.transpose())

# Creating mock dataset AB
# Matching the input for the func correlated_gaussian
#meanAB_=meanAB.flatten().tolist()
#covAB_=covAB.tolist()

# Calculate logLmaxAB using eqn 18 from paper
def find_logLmaxAB(meanA,meanB,covA,covB,logLmaxA=0, logLmaxB=0):
    meandiff = np.array(meanA)-np.array(meanB)
    return -1/2 * meandiff@linalg.solve(covA+covB,meandiff)+logLmaxA+logLmaxB

logLmaxAB = find_logLmaxAB(meanA,meanB,covA,covB,logLmaxA, logLmaxB)

samplesAB = correlated_gaussian(nlive, meanAB, covAB,bounds, logLmaxAB)
samplesAB.plot_2d(axes)
#%%
# Set parameters
nsamples = 1000
beta = 1

def tension_stats(A,B,AB,nsamples):
    # Compute Nested Sampling statistics
    statsA = A.stats(nsamples,beta)
    statsB = B.stats(nsamples,beta)
    statsAB = AB.stats(nsamples,beta)

    samples = Samples(index=statsA.index)
    logR = statsAB.logZ-statsA.logZ-statsB.logZ
    samples['logR']=logR
    logI  = statsA.D_KL + statsB.D_KL - statsAB.D_KL
    samples['logI']=logI
    samples['logS']=logR-logI
    samples['d_G']=statsA.d_G + statsB.d_G - statsAB.d_G
    #samples['logL_P']=statsAB.logL_P-statsA.logL_P-statsB.logL_P
    return samples

samples = tension_stats(samplesA,samplesB,samplesAB,nsamples)
print(samples)
axes = samples.plot_2d(['logR','logI','logS','d_G'])
#%%
# Another way to make corner plots
params = ['logR','logI','logS','d_G']
fig, axes = make_2d_axes(params, figsize=(6, 6), facecolor='w', upper=False)
samples.plot_2d(axes)#, label="model 1")
# %%
########### Using exact solution ############
# Calculate the exact solution of logR
#meandiff = np.array(meanA)-np.array(meanB)

def get_logR(meanA,meanB,covA,covB,V):
    #return -1/2*np.asmatrix(meanA-meanB)*linalg.inv((covA+covB))*(meanA-meanB)-1/2*np.log(abs(2*np.pi*(covA+covB)))+np.log(V)
    #return -1/2*np.asmatrix(meanA-meanB)@linalg.inv((covA+covB))@(np.asmatrix((meanA-meanB)).transpose())-1/2*np.log(linalg.det(2*np.pi*(covA+covB)))+np.log(V)
    meandiff = np.array(meanA)-np.array(meanB)
    return -1/2*meandiff@linalg.solve(covA+covB,meandiff)-1/2*np.linalg.slogdet(2*np.pi*(covA+covB))[1]+np.log(V)


V=1
#meanA_ = np.array(meanA)
#meanB_ = np.array(meanB)
logR_exact = get_logR(meanA,meanB,covA,covB,V)
print(logR_exact)
samples.logR.plot.hist()
plt.axvline(logR_exact,color='r',label='Exact solution')
plt.legend()
plt.show()
# %%
def get_logS(meanA,meanB,covA,covB,d):
    meandiff = np.array(meanA)-np.array(meanB)
    return d/2 - 1/2*meandiff@linalg.solve(covA+covB,meandiff)
d=3
logS_exact = get_logS(meanA,meanB,covA,covB,d)
print(logS_exact)
samples.logS.plot.hist()
plt.axvline(logS_exact,color='r',label='Exact solution')
plt.legend()
plt.show()
# %%
def get_logI(covA,covB,d,V):
    return -d/2 -1/2* np.log(linalg.det(2*np.pi*(covA+covB))) + np.log(V)
logI_exact = get_logI(covA,covB,d,V)
print(logI_exact)
samples.logI.plot.hist()
plt.axvline(logI_exact,color='r',label='Exact solution')
plt.legend()
plt.show()
# %%
