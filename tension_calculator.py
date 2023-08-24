class tension_calculator(dict):
     """Class for computing tension metrics

     Parameters
     ----------
     samples : dict
         Dictionary of NestedSamples
     nsamples : int
         Number of samples to use for computing statistics
     """
     def __init__(self, samples, nsamples=1000):
         for k, v in tqdm.tqdm(samples.items()):
             self[k] = v.stats(nsamples)

     def union(self, A, B):
         A_ = set(A.split('_'))
         B_ = set(B.split('_'))
         return  '_'.join(sorted(list(A_.union(B_))))

     def intersection(self, A, B):
         A_ = set(A.split('_'))
         B_ = set(B.split('_'))
         return  '_'.join(sorted(list(A_.intersection(B_))))

     def stat(self, A, B, stat):
         union = self.union(A, B)
         intersection = self.intersection(A, B)
         return self[union][stat] - self[A][stat] - self[B][stat] + self[intersection][stat]

     def logR(self, A, B):
         return self.stat(A, B, 'logZ')

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
         return scipy.stats.chi2.sf(d[d>0]-2*logS[d>0], d[d>0])