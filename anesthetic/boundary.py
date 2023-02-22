"""Boundary correction utilities."""

import numpy as np
from scipy.special import erf


def cut_and_normalise_gaussian(x, p, bw, xmin=None, xmax=None):
    """Cut and normalise boundary correction for a Gaussian kernel.

    Parameters
    ----------
    x : array-like
        locations for normalisation correction

    p : array-like
        probability densities for normalisation correction

    bw : float
        bandwidth of KDE

    xmin, xmax : float
        lower/upper prior bound
        optional, default None

    Returns
    -------
    p : np.array
        corrected probabilities

    """
    correction = np.ones_like(x)

    if xmin is not None:
        correction *= 0.5 * (1 + erf((x-xmin)/bw/np.sqrt(2)))
        correction[x < xmin] = np.inf
    if xmax is not None:
        correction *= 0.5 * (1 + erf((xmax-x)/bw/np.sqrt(2)))
        correction[x > xmax] = np.inf
    return p/correction
