import anesthetic.examples.perfect_ns 
import numpy as np

params = ['logZ', 'D_KL', 'logL_P', 'd_G']
stats = anesthetic.examples.perfect_ns.correlated_gaussian(100,0,np.array([1,1],[1,1])) 