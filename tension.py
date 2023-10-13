import numpy as np
import scipy as sp
class TensionCalculator:
    def __init__(self,A,B,AB,nsamples):
        self.nsamples = nsamples # draw from random compress t
        self.A = A.stats(nsamples)
        self.B = B.stats(nsamples)
        self.AB = AB.stats(nsamples)

def logR(self):
    return self.AB.logZ-self.A.logZ-self.B.logZ

def logS(self, A, B):
    return self.stat(A, B, 'logL_P')

def d(self, A, B):
    return self.stat(A, B, 'd_G')

def D_KL(self, A, B):
    return self.stat(A, B, 'D_KL')

def p(self, A, B, d=None):
    logS = self.logS(A, B)
    if d is None:
        d = self.d(A, B)
    else:
        d = np.ones_like(logS)*d
    return sp.stats.chi2.sf(d[d>0]-2*logS[d>0], d[d>0])