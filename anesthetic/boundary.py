"""Boundary correction utilities.

Implements boundary correction for Gaussian kernel density estimates following
M. C. Jones (1993) "Simple boundary correction for kernel density estimation",
Statistics and Computing, 3, 135-146.
"""

import numpy as np
from scipy.stats import norm, multivariate_normal

gaussian = norm(loc=0, scale=1)


def _truncated_moments(x, bw, xmin=None, xmax=None):
    r"""Compute truncated Gaussian kernel moments a_0, a_1, a_2.

    For a standard Gaussian kernel K(u), the j-th moment truncated to the
    domain [xmin, xmax] is:

    .. math::
        a_j(x) = \int_{-q_{lo}}^{q_{hi}} u^j K(u) du

    where q_lo = (x - xmin)/bw and q_hi = (xmax - x)/bw.

    Parameters
    ----------
    x : np.array
        Evaluation points.
    bw : float
        Bandwidth.
    xmin, xmax : float, optional
        Lower/upper prior bounds.

    Returns
    -------
    a0, a1, a2 : np.array
        Zeroth, first, and second truncated kernel moments.

    """
    a0 = np.ones_like(x, dtype=float)
    a1 = np.zeros_like(x, dtype=float)
    a2 = np.ones_like(x, dtype=float)

    if xmin is not None:
        q_lo = (x - xmin) / bw
        # Subtract the lower tail: integral from -inf to -q_lo
        a0 -= gaussian.cdf(-q_lo)
        a1 -= gaussian.pdf(q_lo)
        a2 -= gaussian.cdf(-q_lo) + q_lo * gaussian.pdf(q_lo)

    if xmax is not None:
        q_hi = (xmax - x) / bw
        # Subtract the upper tail: integral from q_hi to inf
        a0 -= gaussian.cdf(-q_hi)
        a1 += gaussian.pdf(q_hi)
        a2 -= gaussian.cdf(-q_hi) + q_hi * gaussian.pdf(q_hi)

    return a0, a1, a2


def _kde_eval(kde, x, cov):
    r"""Evaluate KDE and per-axis gradient-weighted KDEs in a single pass.

    For a KDE with kernel K and samples {s_i} with weights {w_i}, computes:

    .. math::
        f(x)    &= \sum_i w_i \, K(x - s_i) \\
        f'_k(x) &= \sum_i w_i \, \frac{x_k - s_{i,k}}{h_k} \, K(x - s_i)

    Parameters
    ----------
    kde : scipy.stats.gaussian_kde
        Fitted KDE object (1D or 2D).
    x : np.array
        Evaluation coordinates, shape (d, M).
    cov : np.array
        Kernel covariance matrix, shape (d, d).

    Returns
    -------
    f : np.array
        KDE density, shape (M,).
    f_prime : tuple of np.array
        Gradient-weighted densities per axis, each shape (M,).

    """
    bw = np.sqrt(np.diag(cov))
    d = x.shape[0]
    data = kde.dataset  # (d, N)
    weights = kde.weights  # (N,)

    diff = x[:, :, None] - data[:, None, :]  # (d, M, N)

    mvn = multivariate_normal(mean=np.zeros(d), cov=cov, allow_singular=True)
    kernel = mvn.pdf(diff.T)  # (N, M)
    wk = weights[:, None] * kernel  # (N, M)

    f = wk.sum(axis=0)
    f_prime = tuple((wk * diff[ax].T / bw[ax]).sum(axis=0) for ax in range(d))
    return f, f_prime


def _corrected_density(f, f_prime, a0, a1, a2, order):
    """Compute boundary-corrected density from KDE and truncated moments.

    Implements renormalization (order 0) and linear boundary correction
    (order 1) from Jones (1993), Eq. (3.4).
    https://doi.org/10.1007/BF00147776

    Parameters
    ----------
    f : np.array
        Uncorrected KDE density.
    f_prime : np.array
        Gradient-weighted KDE density.
    a0, a1, a2 : np.array
        Truncated kernel moments.
    order : int
        Boundary correction order (0 or 1).

    Returns
    -------
    p : np.array
        Corrected density (non-negative).

    """
    p = np.zeros_like(f)
    if order == 0:
        mask = a0 > 0
        p[mask] = f[mask] / a0[mask]
    elif order == 1:
        denominator = a0 * a2 - a1**2
        mask = denominator > 0
        p[mask] = (a2 * f - a1 * f_prime)[mask] / denominator[mask]
    else:
        raise ValueError(f"order must be 0 or 1, got {order}")

    return p


def boundary_correction_gaussian(kde, x, cov, xmin=None, xmax=None, order=1):
    r"""Boundary correction for a 1D Gaussian KDE.

    Parameters
    ----------
    kde : scipy.stats.gaussian_kde
        Fitted 1D KDE object.
    x : np.array
        Evaluation points.
    cov : np.array
        Kernel covariance matrix, shape (1, 1).
    xmin, xmax : float, optional
        Lower/upper bounds.
    order : int, default=1
        Boundary correction order.

        * < 0: no correction --- return raw KDE estimate.
        * 0: renormalization --- O(h) bias.
        * 1: linear correction (Jones 1993, Eq. 3.4) --- O(h²) bias.

    Returns
    -------
    p : np.array
        Boundary-corrected density values.

    """
    if xmin is None and xmax is None or order < 0:
        return kde(x)

    bw = np.sqrt(np.diag(cov))
    f, (f_prime,) = _kde_eval(kde, x[None, :], cov)
    a0, a1, a2 = _truncated_moments(x, bw, xmin, xmax)

    p = _corrected_density(f, f_prime, a0, a1, a2, order)

    if xmin is not None:
        p[x < xmin] = 0
    if xmax is not None:
        p[x > xmax] = 0

    return p


def boundary_correction_gaussian_2d(kde, X, Y, cov,
                                    xmin=None, xmax=None,
                                    ymin=None, ymax=None, order=1):
    r"""Separable boundary correction for a 2D Gaussian KDE.

    Applies the 1D boundary correction independently per axis as a
    multiplicative correction factor.

    Parameters
    ----------
    kde : scipy.stats.gaussian_kde
        Fitted 2D KDE object.
    X, Y : np.array
        2D evaluation grids (from np.mgrid).
    cov : np.array
        Kernel covariance matrix, shape (2, 2).
    xmin, xmax, ymin, ymax : float, optional
        Bounds per axis.
    order : int, default=1
        Boundary correction order.

        * < 0: no correction --- return raw KDE estimate.
        * 0: renormalization --- O(h) bias.
        * 1: linear correction (Jones 1993, Eq. 3.4) --- O(h²) bias.

    Returns
    -------
    p : np.array
        Boundary-corrected 2D density, same shape as X and Y.

    """
    x = X.ravel()
    y = Y.ravel()

    has_x_bounds = xmin is not None or xmax is not None
    has_y_bounds = ymin is not None or ymax is not None

    if not has_x_bounds and not has_y_bounds or order < 0:
        return kde([x, y]).reshape(X.shape)

    bw = np.sqrt(np.diag(cov))
    f, (f_x, f_y) = _kde_eval(kde, np.vstack([x, y]), cov)

    p = np.ones_like(f)
    for has_bounds, bw_ax, f_ax, coords, lo, hi in [
            (has_x_bounds, bw[0], f_x, x, xmin, xmax),
            (has_y_bounds, bw[1], f_y, y, ymin, ymax)
    ]:
        if has_bounds:
            a0, a1, a2 = _truncated_moments(coords, bw_ax, lo, hi)
            p *= _corrected_density(f, f_ax, a0, a1, a2, order)

    # The product of two corrected 1d densities `p_x * p_y` double counts the
    # base density `f` --> Divide product by `f`:
    if has_x_bounds and has_y_bounds:
        mask = f > 0
        p[mask] /= f[mask]
        p[~mask] = 0

    if xmin is not None:
        p[x < xmin] = 0
    if xmax is not None:
        p[x > xmax] = 0
    if ymin is not None:
        p[y < ymin] = 0
    if ymax is not None:
        p[y > ymax] = 0

    return p.reshape(X.shape)
