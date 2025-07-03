"""Nested sampling termination criteria."""
from scipy.special import logsumexp
import numpy as np


def logZ(samples, eps=1e-3, nsamples=None, beta=None):
    """Terminate when evidence in the live points is fraction eps of total."""
    i_live = samples.live_points().index
    logw = samples.logw(nsamples, beta)
    logZ_live = logsumexp(logw.loc[i_live])
    logZ = logsumexp(logw, axis=0)
    return logZ_live - logZ < np.log(eps)


def D_KL(samples, eps=1e-3, nsamples=None, beta=None):
    """Terminate when D_KL in the live points is fraction eps of total."""
    i_live = samples.live_points().index
    logw = samples.logw(nsamples, beta)
    logZ = logsumexp(logw)
    betalogL = samples._betalogL(beta)
    S = (logw*0).add(betalogL, axis=0) - logZ
    w = np.exp(logw-logZ)
    D_KL_live = (S*w).loc[i_live].sum()
    D_KL = (S*w).sum()
    return D_KL_live / D_KL < eps


def logX(samples, max_logX, nsamples=None):
    """Terminate when the log-volume of the live points reaches a threshold."""
    i_live = samples.live_points().index
    return samples.logX(nsamples).loc[i_live[0]] < max_logX


def ndead(samples, max_ndead):
    """Terminate if the number of dead points exceeds a maximum."""
    return len(samples.dead_points()) > max_ndead


def logL(samples, max_logL):
    """Terminate if the lowest live likelihood exceeds a threshold."""
    return samples.contour() > max_logL
