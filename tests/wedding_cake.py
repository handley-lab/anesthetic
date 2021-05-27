import numpy as np
from scipy.special import logsumexp
from anesthetic import NestedSamples


class WeddingCake():
    """Class for generating samples from a wedding cake 'likelihood'.

    This is a likelihood with nested hypercuboidal plateau regions of constant
    likelihood centered on 0.5, with geometrically decreasing volume by a
    factor of alpha. The value of the likelihood in these plateau regions has a
    gaussian profile with width sigma.

    logL = - alpha^(2 floor(D*log_alpha(2|x-0.5|_infinity))/D) / (8 sigma^2)

    Parameters
    ----------
    D: int
        dimensionality (number of parameters) of the likelihood

    alpha: float
        volume compression between plateau regions

    sigma: float
        width of gaussian profile
    """
    def __init__(self, D=4, alpha=0.5, sigma=0.01):
        self.D = D
        self.alpha = alpha
        self.sigma = sigma

    def logZ(self):
        """Numerically compute the true evidence."""
        mean = np.sqrt(self.D/2.) * (np.log(4*self.D*self.sigma**2)-1
                                     ) / np.log(self.alpha)
        std = -np.sqrt(self.D/2.)/np.log(self.alpha)
        i = np.arange(mean + std*10)

        return logsumexp(-self.alpha**(2*i/self.D)/8/self.sigma**2
                         + i*np.log(self.alpha) + np.log(1-self.alpha))

    def i(self, x):
        """Plateau number of a parameter point."""
        r = np.max(abs(x-0.5), axis=-1)
        return np.floor(self.D*np.log(2*r)/np.log(self.alpha))

    def logL(self, x):
        """Gaussian log-likelihood."""
        ri = self.alpha**(self.i(x)/self.D)/2
        return - ri**2/2/self.sigma**2

    def sample(self, nlive=500):
        """Generate samples from a perfect nested sampling run."""
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

            if samps.iloc[-nlive:].weights.sum()/samps.weights.sum() < 0.001:
                break

        death_likes = np.concatenate([death_likes, live_likes])
        birth_likes = np.concatenate([birth_likes, live_birth_likes])
        points = np.concatenate([points, live_points])

        return NestedSamples(points, logL=death_likes, logL_birth=birth_likes)
