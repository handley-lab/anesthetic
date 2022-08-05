"""Utility functions for nested sampling examples."""
import numpy as np
from scipy.stats import special_ortho_group
from scipy.special import gamma


class random_ellipsoid(object):
    """Draw a point uniformly in an ellipsoid.

    This is defined so that the volume of the ellipsoid is sqrt(det(cov))*V_n.
    where V_n is the volume of the unit n ball, and so that its axes have
    length equal to the square root of the eigenvalues of the covariance
    matrix.

    Parameters
    ----------
    mean: 1d array-like
        The center of mass of the ellipsoid

    cov: 2d array-like
        The covariance structure of the ellipsoid. Axes have lengths equal to
        the square root of the eigenvalues of this matrix.
    """

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def __call__(self, size=None):
        """Generate samples uniformly from the ellipsoid."""
        d = len(self.mean)
        L = np.linalg.cholesky(self.cov)
        x = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=size)
        r = np.linalg.norm(x, axis=-1, keepdims=True)
        u = np.random.power(d, r.shape)
        return self.mean + (u*x/r) @ L.T


def random_covariance(sigmas):
    """Draw a randomly oriented covariance matrix with axes length sigmas.

    Parameters
    ----------
    sigmas, 1d array like
        Lengths of the axes of the ellipsoid.
    """
    d = len(sigmas)
    R = special_ortho_group.rvs(d)
    return R @ np.diag(sigmas) @ np.diag(sigmas) @ R.T


def volume_n_ball(n, r=1):
    """Volume of an n dimensional ball, radius r."""
    return np.pi**(n/2)/gamma(n/2+1)*r**n
