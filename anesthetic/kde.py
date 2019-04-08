"""Kernel density estimation tools.

These act as a wrapper around fastKDE, but could be replaced in future by
alternative kernel density estimators
"""
import warnings
from fastkde import fastKDE
from anesthetic.utils import check_bounds, mirror_1d, mirror_2d


def kde_1d(d, xmin=None, xmax=None):
    """Perform a one-dimensional kernel density estimation.

    Wrapper round fastkde.fastKDE. Boundary corrections implemented by
    reflecting boundary conditions.

    Parameters
    ----------
    d: numpy.array
        Data to perform kde on

    xmin, xmax: float
        lower/upper prior bounds
        optional, default None

    Returns
    -------
    x: numpy.array
        x-coordinates of kernel density estimates
    p: numpy.array
        kernel density estimates

    """
    xmin, xmax = check_bounds(d, xmin, xmax)
    f = xmax is None or xmin is None
    d_ = mirror_1d(d, xmin, xmax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, x = fastKDE.pdf(d_, axisExpansionFactor=f,
                           numPointsPerSigma=10*(2-f))
    p *= 2-f

    if xmin is not None:
        p = p[x >= xmin]
        x = x[x >= xmin]

    if xmax is not None:
        p = p[x <= xmax]
        x = x[x <= xmax]

    return x, p


def kde_2d(d_x, d_y, xmin=None, xmax=None, ymin=None, ymax=None):
    """Perform a two-dimensional kernel density estimation.

    Wrapper round fastkde.fastKDE. Boundary corrections implemented by
    reflecting boundary conditions.

    Parameters
    ----------
    d_x, d_y: numpy.array
        x/y coordinates of data to perform kde on

    xmin, xmax, ymin, ymax: float
        lower/upper prior bounds in x/y coordinates
        optional, default None

    Returns
    -------
    x,y: numpy.array
        x/y-coordinates of kernel density estimates. One-dimensional array
    p: numpy.array
        kernel density estimates. Two-dimensional array

    """
    xmin, xmax = check_bounds(d_x, xmin, xmax)
    ymin, ymax = check_bounds(d_y, ymin, ymax)
    f = [xmax is None or xmin is None,
         ymax is None or ymin is None]
    d_x_, d_y_ = mirror_2d(d_x, d_y, xmin, xmax, ymin, ymax)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, (x, y) = fastKDE.pdf(d_x_, d_y_, axisExpansionFactor=f,
                                numPointsPerSigma=10*(2-f[0])*(2-f[1]))

    p *= (2-f[0])
    p *= (2-f[1])
    if xmin is not None:
        p = p[:, x >= xmin]
        x = x[x >= xmin]

    if xmax is not None:
        p = p[:, x <= xmax]
        x = x[x <= xmax]

    if ymin is not None:
        p = p[y >= ymin, :]
        y = y[y >= ymin]

    if ymax is not None:
        p = p[y <= ymax, :]
        y = y[y <= ymax]

    return x, y, p
