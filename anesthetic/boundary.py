"""Boundary correction utilities.

Implements boundary correction for Gaussian kernel density estimates following
M. C. Jones (1993) "Simple boundary correction for kernel density estimation",
Statistics and Computing, 3, 135-146.
"""

import numpy as np
from scipy.stats import norm
from scipy.stats._stats import gaussian_kernel_estimate


def _truncated_moments(x, bw, xmin=None, xmax=None):
    r"""Compute truncated Gaussian kernel moments `a0`, `a1`, `a2`.

    For a standard Gaussian kernel `K(u)` with `u = (x-s)/bw`, the j-th
    truncated moment is computed by subtracting the tails that fall outside
    the support `[xmin, xmax]` from the full-line integral:

    .. math::
        a_j(x) = \int_{-\infty}^{\infty} u^j K(u) du
               - \int_{q_{lo}}^{\infty} u^j K(u) du
               - \int_{-\infty}^{-q_{hi}} u^j K(u) du

    where `q_lo = (x-xmin)/bw` and `q_hi = (xmax-x)/bw`. The second and third
    terms are only subtracted when `xmin` and `xmax` are set, respectively.

    Parameters
    ----------
    x : np.array
        Evaluation points.
    bw : float | np.floating
        Bandwidth.
    xmin, xmax : float, optional
        Lower/upper prior bounds.

    Returns
    -------
    a0, a1, a2 : np.array
        Zeroth, first, and second truncated kernel moments.

    """
    gaussian = norm(loc=0, scale=1)

    a0 = np.ones_like(x, dtype=float)
    a1 = np.zeros_like(x, dtype=float)
    a2 = np.ones_like(x, dtype=float)

    if xmin is not None:
        q_lo = (x - xmin) / bw
        # Subtract: int_{q_lo}^{inf} u^j K(u) du
        a0 -= gaussian.cdf(-q_lo)
        a1 -= gaussian.pdf(-q_lo)
        a2 -= gaussian.cdf(-q_lo) + q_lo * gaussian.pdf(q_lo)

    if xmax is not None:
        q_hi = (xmax - x) / bw
        # Subtract: int_{-inf}^{-q_hi} u^j K(u) du
        a0 -= gaussian.cdf(-q_hi)
        a1 += gaussian.pdf(-q_hi)
        a2 -= gaussian.cdf(-q_hi) + q_hi * gaussian.pdf(q_hi)

    return a0, a1, a2


def _kde_eval(kde, x):
    r"""Evaluate KDE and first bw-scaled moment per-axis in a single pass.

    For a KDE with kernel K and samples `s_i` with weights `w_i`, computes:

    .. math::
        f(x)       = m_{0,k}(x) &= \sum_i w_i K(x - s_i) \\
        -f_k'(x/h) = m_{1,k}(x) &= \sum_i w_i \frac{x_k-s_{i,k}}{h_k} K(x-s_i)

    Parameters
    ----------
    kde : scipy.stats.gaussian_kde
        Fitted KDE object (1D or 2D).
    x : np.array
        Evaluation coordinates, shape (d, M).

    Returns
    -------
    f : np.array
        KDE density, shape (M,).
        Equivalently the zeroth bandwidth-scaled kernel moment.
    moment1 : np.array
        First bandwidth-scaled kernel moment per axis, shape (d, M).
        Equivalently, the negative KDE gradient in bandwidth-scaled
        coordinates, moment1_k = -∂f / ∂(x_k / h_k).

    """
    # some type info needed for Cython
    output_dtype = np.common_type(kde.covariance, x)
    spec_dict = {4: 'float', 8: 'double', 12: 'long double', 16: 'long double'}
    spec = spec_dict[np.dtype(output_dtype).itemsize]

    weighted_fields = np.empty((kde.n, kde.d + 1), dtype=output_dtype)
    weighted_fields[:, 0] = kde.weights
    weighted_fields[:, 1:] = kde.weights[:, None] * kde.dataset.T
    estimate = gaussian_kernel_estimate[spec](kde.dataset.T,
                                              weighted_fields,
                                              x.T,
                                              kde.cho_cov,
                                              output_dtype)
    f = estimate[:, 0]  # (M,)
    bw = np.sqrt(np.diag(kde.covariance))[:, None]  # (d, 1)
    moment1 = (x * f[None, :] - estimate[:, 1:].T) / bw  # (d, M)

    return f, moment1


def _corrected_density(f, moment1, a0, a1, a2, order):
    """Compute boundary-corrected density from KDE and truncated moments.

    Implements renormalization (order 0) and linear boundary correction
    (order 1) from Jones (1993), Eq. (3.4).
    https://doi.org/10.1007/BF00147776

    Parameters
    ----------
    f : np.array
        Uncorrected KDE density, shape (M,).
    moment1 : np.array
        First bandwidth-scaled kernel moment per axis, shape (d, M).
        Equivalently, the negative KDE gradient in bandwidth-scaled
        coordinates, moment1_k = -∂f / ∂(x_k / h_k).
    a0, a1, a2 : np.array
        Truncated kernel moments.
    order : int
        Boundary correction order (0 or 1).

    Returns
    -------
    p : np.array
        Corrected density, clamped to non-negative values. The linear
        correction (order 1) can produce negative values near boundaries
        (Jones 1993, p. 142); these are truncated to zero.

    """
    p = np.zeros_like(f)
    if order == 0:
        mask = a0 > 0
        p[mask] = f[mask] / a0[mask]
    elif order == 1:
        denominator = a0 * a2 - a1**2
        mask = denominator > 0
        p[mask] = (a2 * f - a1 * moment1)[mask] / denominator[mask]
    else:
        raise ValueError(f"order must be 0 or 1, got {order}")

    np.maximum(p, 0, out=p)
    return p


def boundary_correction_1d(kde, x, order=1, xmin=None, xmax=None):
    r"""Boundary correction for a 1D Gaussian KDE.

    Parameters
    ----------
    kde : scipy.stats.gaussian_kde
        Fitted 1D KDE object.
    x : np.array
        Evaluation points.
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

    bw = np.sqrt(kde.covariance[0, 0])
    f, (moment1,) = _kde_eval(kde, x[None, :])
    a0, a1, a2 = _truncated_moments(x, bw, xmin, xmax)

    p = _corrected_density(f, moment1, a0, a1, a2, order)

    if xmin is not None:
        p[x < xmin] = 0
    if xmax is not None:
        p[x > xmax] = 0

    return p


def boundary_correction_2d(kde, X, Y, order=1,
                           xmin=None, xmax=None, ymin=None, ymax=None):
    r"""Separable boundary correction for a 2D Gaussian KDE.

    Applies the 1D boundary correction independently per axis as a
    multiplicative correction factor.

    Parameters
    ----------
    kde : scipy.stats.gaussian_kde
        Fitted 2D KDE object.
    X, Y : np.array
        2D evaluation grids (from np.mgrid).
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

    bw_x = np.sqrt(kde.covariance[0, 0])
    bw_y = np.sqrt(kde.covariance[1, 1])
    f, (moment1_x, moment1_y) = _kde_eval(kde, np.vstack([x, y]))

    p = np.ones_like(f)
    for has_bounds, bw_ax, moment1_ax, coords, lo, hi in [
            (has_x_bounds, bw_x, moment1_x, x, xmin, xmax),
            (has_y_bounds, bw_y, moment1_y, y, ymin, ymax)
    ]:
        if has_bounds:
            a0, a1, a2 = _truncated_moments(coords, bw_ax, lo, hi)
            p *= _corrected_density(f, moment1_ax, a0, a1, a2, order)

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
