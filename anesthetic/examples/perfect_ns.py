"""Perfect nested sampling generators."""
import numpy as np
from scipy.stats import multivariate_normal
from anesthetic.examples.utils import random_ellipsoid
from anesthetic import NestedSamples
from anesthetic.samples import merge_nested_samples


def gaussian(nlive, ndims, sigma=0.1, R=1, logLmin=-1e-2):
    """Perfect nested sampling run for a spherical Gaussian & prior.

    Up to normalisation this is identical to the example in John Skilling's
    original paper https://doi.org/10.1214/06-BA127 . Both spherical Gaussian
    width sigma and spherical uniform prior width R are centered on zero

    Parameters
    ----------
    nlive: int
        number of live points

    ndims: int
        dimensionality of gaussian

    sigma: float
        width of gaussian likelihood

    R: float
        radius of gaussian prior

    logLmin: float
        loglikelihood at which to terminate

    Returns
    -------
    samples: NestedSamples
        Nested sampling run
    """

    def logLike(x):
        return -(x**2).sum(axis=-1)/2/sigma**2

    def random_sphere(n):
        return random_ellipsoid(np.zeros(ndims), np.eye(ndims), n)

    samples = []
    r = R
    logL_birth = np.ones(nlive) * -np.inf
    logL = logL_birth.copy()
    while logL.min() < logLmin:
        points = r * random_sphere(nlive)
        logL = logLike(points)
        samples.append(NestedSamples(points, logL=logL, logL_birth=logL_birth))
        logL_birth = logL.copy()
        r = (points**2).sum(axis=-1, keepdims=True)**0.5

    samples = merge_nested_samples(samples)
    samples.logL
    logLend = samples[samples.nlive >= nlive].logL.max()
    return samples[samples.logL_birth < logLend].recompute()


def correlated_gaussian(nlive, mean, cov):
    """Perfect nested sampling run for a correlated gaussian likelihood.

    This produces a perfect nested sampling run with a uniform prior over the
    unit hypercube, with a likelihood gaussian in the parameters normalised so
    that the evidence is unity. The algorithm proceeds by simultaneously
    rejection sampling from the prior and sampling exactly and uniformly from
    the known ellipsoidal contours.

    This can produce perfect runs in up to around d~15 dimensions. Beyond
    this rejection sampling from a truncated gaussian in the early stage
    becomes too inefficient.

    Parameters
    ----------
    nlive: int
        minimum number of live points across the run

    mean: 1d array-like
        mean of gaussian in parameters. Length of array defines dimensionality
        of run.

    cov: 2d array-like
        covariance of gaussian in parameters

    Returns
    -------
    samples: NestedSamples
        Nested sampling run
    """

    def logLike(x):
        return multivariate_normal(mean, cov).logpdf(x)

    logLmax = logLike(mean)

    points = np.random.rand(2*nlive, len(mean))
    samples = NestedSamples(points, logL=logLike(points), logL_birth=-np.inf)

    while (1/samples.nlive.iloc[:-nlive]).sum() < samples.D()*2:
        logLs = samples.logL.iloc[-nlive]

        # Cube round
        points = np.random.rand(nlive, len(mean))
        logL = logLike(points)
        i = logL > logLs
        samps_1 = NestedSamples(points[i], logL=logL[i], logL_birth=logLs)

        # Ellipsoidal round
        points = random_ellipsoid(mean, cov*2*(logLmax - logLs), nlive)
        logL = logLike(points)
        i = ((points > 0) & (points < 1)).all(axis=1)
        samps_2 = NestedSamples(points[i], logL=logL[i], logL_birth=logLs)

        samples = merge_nested_samples([samples, samps_1, samps_2])

    return samples


def wedding_cake(nlive, ndims, sigma=0.01, alpha=0.5):
    """Perfect nested sampling run for a wedding cake likelihood.

    This is a likelihood with nested hypercuboidal plateau regions of constant
    likelihood centered on 0.5, with geometrically decreasing volume by a
    factor of alpha. The value of the likelihood in these plateau regions has a
    gaussian profile with width sigma.

    logL = - alpha^(2 floor(D*log_alpha(2|x-0.5|_infinity))/D) / (8 sigma^2)

    Parameters
    ----------
    nlive: int
        number of live points

    ndims: int
        dimensionality of the likelihood

    sigma: float
        width of gaussian profile

    alpha: float
        volume compression between plateau regions
    """

    def i(x):
        """Plateau number of a parameter point."""
        r = np.max(abs(x-0.5), axis=-1)
        return np.floor(ndims*np.log(2*r)/np.log(alpha))

    def logL(x):
        """Gaussian log-likelihood."""
        ri = alpha**(i(x)/ndims)/2
        return - ri**2/2/sigma**2

    points = np.zeros((0, ndims))
    death_likes = np.zeros(0)
    birth_likes = np.zeros(0)

    live_points = np.random.rand(nlive, ndims)
    live_likes = logL(live_points)
    live_birth_likes = np.ones(nlive)*-np.inf

    while True:
        logL_ = live_likes.min()
        j = live_likes == logL_

        death_likes = np.concatenate([death_likes, live_likes[j]])
        birth_likes = np.concatenate([birth_likes, live_birth_likes[j]])
        points = np.concatenate([points, live_points[j]])

        i_ = i(live_points[j][0])+1
        r_ = alpha**(i_/ndims)/2
        x_ = np.random.uniform(0.5-r_, 0.5+r_, size=(j.sum(), ndims))
        live_birth_likes[j] = logL_
        live_points[j] = x_
        live_likes[j] = logL(x_)

        samps = NestedSamples(points, logL=death_likes, logL_birth=birth_likes)

        if samps.iloc[-nlive:].weights.sum()/samps.weights.sum() < 0.001:
            break

    death_likes = np.concatenate([death_likes, live_likes])
    birth_likes = np.concatenate([birth_likes, live_birth_likes])
    points = np.concatenate([points, live_points])

    return NestedSamples(points, logL=death_likes, logL_birth=birth_likes)
