"""Kernel density estimation tools.

These act as a wrapper around fastKDE, but could be replaced in future by
alternative kernel density estimators
"""
import warnings
from fastkde import fastKDE
from anesthetic.utils import mirror_1d, mirror_2d


def fastkde_1d(d, xmin=None, xmax=None):
    """Perform a one-dimensional kernel density estimation.

    Wrapper around `fastkde.fastKDE <https://github.com/LBL-EESA/fastkde>`_.
    Boundary corrections implemented by reflecting boundary conditions.

    Parameters
    ----------
    d : np.array
        Data to perform kde on

    xmin, xmax : float
        lower/upper prior bounds
        optional, default None

    Returns
    -------
    x : np.array
        x-coordinates of kernel density estimates
    p : np.array
        kernel density estimates

    """
    f = xmax is None or xmin is None
    d_ = mirror_1d(d, xmin, xmax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, x = fastKDE.pdf(d_, axis_expansion_factor=f,
                           num_points_per_sigma=10*(2-f),
                           use_xarray=False)
    p *= 2-f

    if xmin is not None:
        p = p[x >= xmin]
        x = x[x >= xmin]

    if xmax is not None:
        p = p[x <= xmax]
        x = x[x <= xmax]

    return x, p, xmin, xmax


def fastkde_2d(d_x, d_y, xmin=None, xmax=None, ymin=None, ymax=None):
    """Perform a two-dimensional kernel density estimation.

    Wrapper round `fastkde.fastKDE <https://github.com/LBL-EESA/fastkde>`_.
    Boundary corrections implemented by reflecting boundary conditions.

    Parameters
    ----------
    d_x, d_y : np.array
        x/y coordinates of data to perform kde on

    xmin, xmax, ymin, ymax : float
        lower/upper prior bounds in x/y coordinates
        optional, default None

    Returns
    -------
    x, y : np.array
        x/y-coordinates of kernel density estimates. One-dimensional array
    p : np.array
        kernel density estimates. Two-dimensional array

    """
    f = [xmax is None or xmin is None,
         ymax is None or ymin is None]
    d_x_, d_y_ = mirror_2d(d_x, d_y, xmin, xmax, ymin, ymax)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, (x, y) = fastKDE.pdf(d_x_, d_y_, axis_expansion_factor=f,
                                num_points_per_sigma=10*(2-f[0])*(2-f[1]),
                                use_xarray=False)

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

    return x, y, p, xmin, xmax, ymin, ymax
