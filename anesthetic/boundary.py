"""Boundary correction utilities."""

import numpy as np
from scipy.special import erf


def cut_and_normalise_gaussian(x, p, sigma, xmin=None, xmax=None):
    """Cut and normalise boundary correction for a Gaussian kernel.

    Parameters
    ----------
    x: np.array
        locations for normalisation correction

    p: np.array
        probability densities for normalisation correction

    sigma: float
        bandwidth of KDE

    xmin, xmax: float
        lower/upper prior bound
        optional, default None

    Returns
    -------
    p: np.array
        corrected probabilities

    """
    correction = np.ones_like(x)

    if xmin is not None:
        correction *= 0.5*(1 + erf((x - xmin)/sigma/np.sqrt(2)))
        correction[x < xmin] = np.inf
    if xmax is not None:
        correction *= 0.5*(1 + erf((xmax - x)/sigma/np.sqrt(2)))
        correction[x > xmax] = np.inf
    return p/correction
