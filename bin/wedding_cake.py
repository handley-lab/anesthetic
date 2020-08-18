import numpy as np
from scipy.special import logsumexp
from numpy.testing import assert_array_less
from anesthetic import NestedSamples
import matplotlib.pyplot as plt

np.random.seed(0)
D=4
alpha = 0.5
sigma = 0.01
nlive = 500

i = np.arange(1000)
ri = alpha**(i/D)/2
logZ = logsumexp(-ri**2/2/sigma**2 + i*np.log(alpha) + np.log(1-alpha))   

def r(x):
    return np.max(abs(x-0.5),axis=-1)

def i(x):
    return np.floor(D*np.log(2*r(x))/np.log(alpha)) 

def logL(x):
    ri = alpha**(i(x)/D)/2
    return - ri**2/2/sigma**2


points = np.zeros((0,D))
death_likes = np.zeros(0)
birth_likes = np.zeros(0)

live_points = np.random.rand(nlive, D)
live_likes = logL(live_points)
live_birth_likes = np.ones(nlive)*-np.inf

while True:
    logL_ = live_likes.min() 
    j = live_likes == logL_

    death_likes = np.concatenate([death_likes, live_likes[j]])
    birth_likes = np.concatenate([birth_likes, live_birth_likes[j]]) 
    points = np.concatenate([points, live_points[j]])

    i_ = i(live_points[j][0])+1
    r_ = alpha**(i_/D)/2
    x_ = np.random.uniform(0.5-r_,0.5+r_, size=(j.sum(),D))
    live_birth_likes[j] = logL_
    live_points[j] = x_
    live_likes[j] = logL(x_)

    samples = NestedSamples(points, logL=death_likes, logL_birth=birth_likes)
    if samples.live_points().weights.sum()/samples.weights.sum() < 0.001:
        break

death_likes = np.concatenate([death_likes, live_likes])
birth_likes = np.concatenate([birth_likes, live_birth_likes]) 
points = np.concatenate([points, live_points])

samples = NestedSamples(points, logL=death_likes, logL_birth=birth_likes)

root = './tests/example_data/wedding_cake'
samples.loc[:,:'logL_birth'].to_csv(root + '_dead-birth.txt', sep=' ', index=False, header=False)
