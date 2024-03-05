"""Perfect nested sampling generators."""
import numpy as np
from anesthetic.examples.utils import random_ellipsoid
from anesthetic import NestedSamples
from anesthetic.samples import merge_nested_samples


def gaussian(nlive, ndims, sigma=0.1, R=1, logLmin=-1e-2, logLmax=0,
             *args, **kwargs):
    """Perfect nested sampling run for a spherical Gaussian & prior.

    Up to normalisation this is identical to the example in John Skilling's
    original paper https://doi.org/10.1214/06-BA127 . Both spherical Gaussian
    width sigma and spherical uniform prior width R are centered on zero

    Parameters
    ----------
    nlive : int
        number of live points

    ndims : int
        dimensionality of gaussian

    sigma : float
        width of gaussian likelihood

    R : float
        radius of gaussian prior

    logLmin : float
        loglikelihood at which to terminate

    logLmax : float
        maximum loglikelihood

    The remaining arguments are passed to the
    :class:`anesthetic.samples.NestedSamples` constructor.

    Returns
    -------
    samples : :class:`anesthetic.samples.NestedSamples`
        Nested sampling run
    """

    def logLike(x):
        return logLmax - (x**2).sum(axis=-1)/2/sigma**2

    def random_sphere(n):
        return random_ellipsoid(np.zeros(ndims), np.eye(ndims), n)

    samples = []
    r = R
    logL_birth = np.ones(nlive) * -np.inf
    logL = logL_birth.copy()
    while logL.min() < logLmin:
        points = r * random_sphere(nlive)
        logL = logLike(points)
        samples.append(NestedSamples(points, logL=logL, logL_birth=logL_birth,
                                     *args, **kwargs))
        logL_birth = logL.copy()
        r = (points**2).sum(axis=-1, keepdims=True)**0.5

    samples = merge_nested_samples(samples)
    logLend = samples.loc[samples.nlive >= nlive].logL.max()
    return samples.loc[samples.logL_birth < logLend].recompute()


def correlated_gaussian(nlive, mean, cov, bounds=None, logLmax=0,
                        *args, **kwargs):
    """Perfect nested sampling run for a correlated gaussian likelihood.

    This produces a perfect nested sampling run with a uniform prior over
    the unit hypercube. The algorithm proceeds by simultaneously rejection
    sampling from the prior and sampling exactly and uniformly from the
    known ellipsoidal contours.

    This can produce perfect runs in up to around d~15 dimensions. Beyond
    this rejection sampling from a truncated gaussian in the early stage
    becomes too inefficient.

    Parameters
    ----------
    nlive : int
        minimum number of live points across the run

    mean : 1d array-like, shape (ndims,)
        mean of gaussian in parameters. Length of array defines dimensionality
        of run.

    cov : 2d array-like, shape (ndims, ndims)
        covariance of gaussian in parameters

    bounds : 2d array-like, shape (ndims, 2)
        bounds of a gaussian, default ``[[0, 1]]*ndims``

    logLmax : float
        maximum loglikelihood

    The remaining arguments are passed to the
    :class:`anesthetic.samples.NestedSamples` constructor.

    Returns
    -------
    samples : :class:`anesthetic.samples.NestedSamples`
        Nested sampling run
    """
    mean = np.array(mean, dtype=float)
    cov = np.array(cov, dtype=float)
    invcov = np.linalg.inv(cov)

    def logLike(x):
        return logLmax - 0.5 * ((x - mean) @ invcov * (x - mean)).sum(axis=-1)

    ndims = len(mean)

    if bounds is None:
        bounds = [[0, 1]]*ndims

    bounds = np.array(bounds, dtype=float)

    points = np.random.uniform(*bounds.T, (2*nlive, ndims))
    samples = NestedSamples(points, logL=logLike(points), logL_birth=-np.inf,
                            *args, **kwargs)

    while (1/samples.nlive.iloc[:-nlive]).sum() < samples.D_KL()*2:
        logLs = samples.logL.iloc[-nlive]

        sig = np.diag(cov*2*(logLmax - logLs))**0.5
        bounds[:, 0] = np.max([bounds[:, 0], mean - sig], axis=0)
        bounds[:, 1] = np.min([bounds[:, 1], mean + sig], axis=0)

        # Cube round
        points = np.random.uniform(*bounds.T, (nlive, ndims))
        logL = logLike(points)
        i = logL > logLs
        samps_1 = NestedSamples(points[i], logL=logL[i], logL_birth=logLs,
                                *args, **kwargs)

        # Ellipsoidal round
        points = random_ellipsoid(mean, cov*2*(logLmax - logLs), nlive)
        logL = logLike(points)
        i = ((points > bounds.T[0]) & (points < bounds.T[1])).all(axis=1)
        samps_2 = NestedSamples(points[i], logL=logL[i], logL_birth=logLs,
                                *args, **kwargs)
        samples = merge_nested_samples([samples, samps_1, samps_2])

    return samples


def wedding_cake(nlive, ndims, sigma=0.01, alpha=0.5, *args, **kwargs):
    """Perfect nested sampling run for a wedding cake likelihood.

    This is a likelihood with nested hypercuboidal plateau regions of constant
    likelihood centered on 0.5, with geometrically decreasing volume by a
    factor of alpha. The value of the likelihood in these plateau regions has a
    gaussian profile with width sigma.

    ::

        logL = - alpha^(2*floor(D*log_alpha(2|x-0.5|_infinity))/D) / (8sigma^2)

    Parameters
    ----------
    nlive : int
        number of live points

    ndims : int
        dimensionality of the likelihood

    sigma : float
        width of gaussian profile

    alpha : float
        volume compression between plateau regions

    The remaining arguments are passed to the
    :class:`anesthetic.samples.NestedSamples` constructor.
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

        samps = NestedSamples(points, logL=death_likes, logL_birth=birth_likes,
                              *args, **kwargs)
        weights = samps.get_weights()
        if weights[-nlive:].sum() < 0.001 * weights.sum():
            break

    death_likes = np.concatenate([death_likes, live_likes])
    birth_likes = np.concatenate([birth_likes, live_birth_likes])
    points = np.concatenate([points, live_points])

    return NestedSamples(points, logL=death_likes, logL_birth=birth_likes,
                         *args, **kwargs)


def planck_gaussian(nlive=500):
    """Gaussian likelihood with Planck-like posterior.

    This is a gaussian likelihood with the same mean, parameter covariance and
    average loglikelihood as the Planck 2018 legacy chains
    ``base/plikHM_TTTEEE_lowl_lowE_lensing``

    Parameters
    ----------
    nlive : int, optional
        number of live points

    Returns
    -------
    samples : :class:`anesthetic.samples.NestedSamples`
        Nested sampling run
    """
    columns = ['omegabh2', 'omegach2', 'theta', 'tau', 'logA', 'ns']
    labels = [r'$\Omega_b h^2$', r'$\Omega_c h^2$', r'$100\theta_{MC}$',
              r'$\tau$', r'${\rm{ln}}(10^{10} A_s)$', r'$n_s$']

    cov = np.array([
        [2.12e-08, -9.03e-08, 1.76e-08, 2.96e-07, 4.97e-07, 2.38e-07],
        [-9.03e-08, 1.39e-06, -1.26e-07, -3.41e-06, -4.15e-06, -3.28e-06],
        [1.76e-08, -1.26e-07, 9.71e-08, 4.30e-07, 7.41e-07, 4.13e-07],
        [2.96e-07, -3.41e-06, 4.30e-07, 5.33e-05, 9.53e-05, 1.05e-05],
        [4.97e-07, -4.15e-06, 7.41e-07, 9.53e-05, 2.00e-04, 1.35e-05],
        [2.38e-07, -3.28e-06, 4.13e-07, 1.05e-05, 1.35e-05, 1.73e-05]])

    mean = np.array([0.02237, 0.1200, 1.04092, 0.0544, 3.044, 0.9649])

    bounds = np.array([
        [5.00e-03, 1.00e-01],
        [1.00e-03, 9.90e-01],
        [5.00e-01, 1.00e+01],
        [1.00e-02, 8.00e-01],
        [1.61e+00, 3.91e+00],
        [8.00e-01, 1.20e+00]])

    logL_mean = -1400.35
    d = len(mean)
    logLmax = logL_mean + d/2
    return correlated_gaussian(nlive, mean, cov, bounds, logLmax,
                               columns=columns, labels=labels)
