import numpy as np
from scipy.special import logsumexp
from anesthetic import NestedSamples


class WeddingCake():
    def __init__(self, D=4, alpha=0.5, sigma=0.01):
        self.D = D
        self.alpha = alpha
        self.sigma = sigma

    def logZ(self):
        i = np.arange(self.i_mean() + self.i_std()*10)
        ri = self.alpha**(i/self.D)/2
        logZ = logsumexp(-ri**2/2/self.sigma**2
                         + i*np.log(self.alpha) + np.log(1-self.alpha))
        return logZ

    def i_mean(self):
        return np.sqrt(self.D/2.) * (np.log(4*self.D*self.sigma**2)-1
                                     ) / np.log(self.alpha)

    def i_std(self):
        return -np.sqrt(self.D/2.)/np.log(self.alpha)

    def r(self, x):
        return np.max(abs(x-0.5), axis=-1)

    def i(self, x):
        return np.floor(self.D*np.log(2*self.r(x))/np.log(self.alpha))

    def logL(self, x):
        ri = self.alpha**(self.i(x)/self.D)/2
        return - ri**2/2/self.sigma**2

    def sample(self, nlive=500):
        points = np.zeros((0, self.D))
        death_likes = np.zeros(0)
        birth_likes = np.zeros(0)

        live_points = np.random.rand(nlive, self.D)
        live_likes = self.logL(live_points)
        live_birth_likes = np.ones(nlive)*-np.inf

        while True:
            logL_ = live_likes.min()
            j = live_likes == logL_

            death_likes = np.concatenate([death_likes, live_likes[j]])
            birth_likes = np.concatenate([birth_likes, live_birth_likes[j]])
            points = np.concatenate([points, live_points[j]])

            i_ = self.i(live_points[j][0])+1
            r_ = self.alpha**(i_/self.D)/2
            x_ = np.random.uniform(0.5-r_, 0.5+r_, size=(j.sum(), self.D))
            live_birth_likes[j] = logL_
            live_points[j] = x_
            live_likes[j] = self.logL(x_)

            samps = NestedSamples(points, logL=death_likes,
                                  logL_birth=birth_likes)
            if samps.live_points().weights.sum()/samps.weights.sum() < 0.001:
                break

        death_likes = np.concatenate([death_likes, live_likes])
        birth_likes = np.concatenate([birth_likes, live_birth_likes])
        points = np.concatenate([points, live_points])

        return NestedSamples(points, logL=death_likes, logL_birth=birth_likes)
