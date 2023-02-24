"""Utility functions for nested sampling examples."""
import numpy as np
from scipy.stats import special_ortho_group
from scipy.special import gamma, gammaln


def random_ellipsoid(mean, cov, size=None):
    """Draw a point uniformly in an ellipsoid.

    This is defined so that the volume of the ellipsoid is
    ``sqrt(det(cov))*V_n``. where ``V_n`` is the volume of the unit n ball,
    and so that its axes have length equal to the square root of the
    eigenvalues of the covariance matrix.

    Parameters
    ----------
    mean : 1d array-like
        The center of mass of the ellipsoid

    cov : 2d array-like
        The covariance structure of the ellipsoid. Axes have lengths equal to
        the square root of the eigenvalues of this matrix.

    size : int or tuple of ints, optional
        Given a shape of, for example, (m,n,k), m*n*k samples are generated,
        and packed in an m-by-n-by-k arrangement. Because each sample is
        N-dimensional, the output shape is (m,n,k,N). If no shape is specified,
        a single (N-D) sample is returned.

    Returns
    -------
    points : array-like
        The drawn samples, of shape size, if that was provided. If not, the
        shape is (N,).
        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional value
        drawn uniformly from the ellipsoid.
    """
    d = len(mean)
    L = np.linalg.cholesky(cov)
    x = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=size)
    r = np.linalg.norm(x, axis=-1, keepdims=True)
    u = np.random.power(d, r.shape)
    return mean + (u*x/r) @ L.T


def random_covariance(sigmas):
    """Draw a randomly oriented covariance matrix with axes length sigmas.

    Parameters
    ----------
    sigmas : 1d array like
        Lengths of the axes of the ellipsoid.

    Returns
    -------
    Covariance matrix : 2d np.array
        shape (len(sigmas), len(sigmas)).
    """
    d = len(sigmas)
    R = special_ortho_group.rvs(d)
    return R @ np.diag(sigmas) @ np.diag(sigmas) @ R.T


def volume_n_ball(n, r=1):
    """Volume of an n dimensional ball, radius r."""
    return np.pi**(n/2)/gamma(1+n/2)*r**n


def log_volume_n_ball(n, r=1):
    """Log-volume of an n dimensional ball, radius r."""
    return np.log(np.pi)*n/2 - gammaln(1+n/2) + np.log(r)*n
