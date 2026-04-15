"""Boundary correction utilities.

Implements local-linear boundary correction for Gaussian kernel density
estimates following:

M. C. Jones (1993),
"Simple boundary correction for kernel density estimation",
Statistics and Computing, 3, 135-146.
https://doi.org/10.1007/BF00147776

J. E. Chacón & T. Duong (2018),
"Multivariate Kernel Smoothing and Its Applications",
Chapman and Hall/CRC, Chapter 4.
https://doi.org/10.1201/9780429485572-4
https://www.researchgate.net/publication/345555871_Modified_density_estimation

"""

import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.stats._stats import gaussian_kernel_estimate

_DTYPE_SPEC = {4: 'float', 8: 'double', 12: 'long double',
               16: 'long double'}


def _truncated_moments(x, cov, x_limits):
    r"""Compute truncated Gaussian kernel moments.

    For a Gaussian kernel with covariance matrix ``H = D R D``, where
    ``D = diag(sqrt(diag(H)))`` and ``R`` is the corresponding correlation
    matrix, compute the truncated moments of the Gaussian kernel over the part
    of the kernel support that remains inside the box ``x_limits``.

    Parameters
    ----------
    x : np.array
        Evaluation coordinates, shape ``(M, d)``.
    cov : np.array
        Kernel covariance matrix, shape ``(d, d)``.
    x_limits : np.array
        Lower and upper prior bounds in physical coordinates, shape ``(2, d)``.
        Use ``-np.inf`` and ``np.inf`` for unbounded directions.

    Returns
    -------
    a0 : np.array
        Zeroth truncated kernel moment, shape ``(M,)``.
    a1 : np.array
        First bandwidth-scaled truncated kernel moment, shape ``(M, d)``.
    A2 : np.array
        Second bandwidth-scaled truncated kernel moment, shape ``(M, d, d)``.

    Notes
    -----
    The returned moments are the generic ``d``-dimensional objects:

    .. math::
        a_0(x) = \int_{\Omega_x} \phi_R(u) du,

        a_1(x) = \int_{\Omega_x} u \, \phi_R(u) du,

        A_2(x) = \int_{\Omega_x} u u^T \phi_R(u) du,

    where ``u = D^{-1}(x-t)`` and ``\Omega_x`` is the allowed region in these
    bandwidth-scaled coordinates. For ``d = 1`` these reduce exactly to the
    scalar moments used in the 1D Jones correction.

    """
    m, d = x.shape
    if d not in (1, 2):
        raise NotImplementedError(f"boundary correction currently supports "
                                  f"only 1D and 2D KDEs, got d={d}")
    if cov.shape != (d, d):
        raise ValueError(f"cov must have shape {(d, d)}, got {cov.shape}")
    if x_limits.shape != (2, d):
        raise ValueError(f"x_limits must have shape {(2, d)}, "
                         f"got {x_limits.shape}")

    eps = np.finfo(float).eps

    # Split covariance into per-axis scales and a correlation matrix.
    sigma = np.sqrt(np.diag(cov))  # (d,)
    corr = cov / np.outer(sigma, sigma)  # (d, d)
    gaussian_1dim = norm(loc=0, scale=1)
    gaussian_ddim = multivariate_normal(mean=np.zeros(d), cov=corr)

    # Lower/upper bounds in bandwidth-scaled coordinates.
    # (M, 2, d)
    u_limits = (x_limits[None, :, :] - x[:, None, :]) / sigma[None, None, :]

    # All 2^d rectangle corners in bandwidth-scaled coordinates.
    # (M, *(2,)*d, d)
    u_corner = np.stack(np.broadcast_arrays(*[
        u_limits[:, :, i].reshape((m,) + tuple(2 if j == i else 1
                                               for j in range(d)))
        for i in range(d)
    ]), axis=-1)

    # Inclusion-exclusion signs for the corner sums.
    signs = np.array([-1, +1])  # (2,)
    for _ in range(d - 1):
        signs = np.multiply.outer(signs, [-1, +1])  # *(2,)*d

    # Multivariate CDF and marginal PDF at each corner.
    cdf = gaussian_ddim.cdf(u_corner)  # (M, *(2,)*d)
    pdf = gaussian_1dim.pdf(u_corner)  # (M, *(2,)*d, d)

    # Corner contributions to the gradient and Hessian of a0.
    # grad_i = phi(u_i) * Phi(u_{-i} | u_i).
    grad = np.empty_like(u_corner)  # (M, *(2,)*d, d)
    hess = np.zeros(u_corner.shape + (d,))  # (M, *(2,)*d, d, d)
    for i in range(d):
        ui = u_corner[..., i]
        inf_i = np.isinf(ui)
        grad[..., i] = pdf[..., i]
        corr_hess = 0
        # For d=2, multiply by the conditional CDF Phi(u_j|u_i) and build the
        # off-diagonal Hessian from the conditional PDF. For d=1 this block is
        # skipped and grad_i = phi(u_i).
        if d == 2:
            j = 1 - i
            rho_ij = np.clip(corr[i, j], -1 + eps, 1 - eps)
            s_ij = np.sqrt(1 - rho_ij**2)
            uj = u_corner[..., j]
            v = (uj - rho_ij * ui) / s_ij
            v = np.where(np.isposinf(uj), +np.inf, v)
            v = np.where(np.isneginf(uj), -np.inf, v)
            inf_i_fin_j = inf_i & np.isfinite(uj)
            if rho_ij == 0:
                v = np.where(inf_i_fin_j, uj / s_ij, v)
            else:
                v = np.where(inf_i_fin_j, -np.sign(rho_ij) * ui, v)
            grad[..., i] *= gaussian_1dim.cdf(v)
            hess[..., i, j] = pdf[..., i] * gaussian_1dim.pdf(v) / s_ij
            corr_hess += corr[i, j] * hess[..., i, j]
        hess[..., i, i] = np.where(inf_i, 0, -ui * grad[..., i]) - corr_hess

    # Signed corner sums give a0 and its derivatives.
    axes = tuple(range(1, d + 1))  # corner axes
    a0 = np.sum(cdf * signs, axis=axes)  # (M,)
    grad_a0 = -np.sum(grad * signs[..., None], axis=axes)  # (M, d)
    hess_a0 = np.sum(hess * signs[..., None, None], axis=axes)  # (M, d, d)

    # Recover the first and second truncated moments from derivatives of a0.
    a1 = -grad_a0 @ corr  # (M, d)
    A2 = a0[:, None, None] * corr[None, :, :]  # (M, d, d)
    A2 += np.einsum('ij,mjk,kl->mil', corr, hess_a0, corr)  # (M, d, d)

    return a0, a1, A2


def _kde_eval(kde, x):
    r"""Evaluate KDE and first bandwidth-scaled kernel moment in a single pass.

    For a KDE with kernel covariance ``H = D R D``, kernel ``K``, and samples
    ``s_i`` with weights ``w_i``, computes

    .. math::
        f(x) = m_0(x) &= \sum_i w_i K(x - s_i), \\
               m_1(x) &= \sum_i w_i D^{-1}(x-s_i) K(x-s_i)
                       = -R \, \nabla_{D^{-1}x} f(x).

    In 1D, this reduces to the usual bandwidth-scaled kernel moment
    ``m_1(x) = \sum_i w_i (x-s_i) K(x-s_i) / h``.

    Parameters
    ----------
    kde : scipy.stats.gaussian_kde
        Fitted KDE object (1D or 2D).
    x : np.array
        Evaluation coordinates, shape ``(M, d)``.

    Returns
    -------
    f : np.array
        KDE density, shape ``(M,)``.
        Equivalently the zeroth bandwidth-scaled kernel moment.
    moment1 : np.array
        First bandwidth-scaled kernel moment, shape ``(M, d)``.
        Equivalently, ``moment1 = -R ∇_{D^{-1}x} f`` in the
        bandwidth-scaled coordinates.

    """
    output_dtype = np.common_type(kde.covariance, x)
    spec = _DTYPE_SPEC[np.dtype(output_dtype).itemsize]

    # Pack the KDE weights and first raw moments into one Cython call.
    # (N, d+1)
    weighted_fields = np.empty((kde.n, kde.d + 1), dtype=output_dtype)
    weighted_fields[:, 0] = kde.weights  # (N,)
    weighted_fields[:, 1:] = kde.weights[:, None] * kde.dataset.T  # (N, d)

    estimate = gaussian_kernel_estimate[spec](kde.dataset.T,
                                              weighted_fields,
                                              x,
                                              kde.cho_cov,
                                              output_dtype)
    f = estimate[:, 0]  # (M,)

    # Convert first raw moments into bandwidth-scaled residual moments.
    residual = x * f[:, None] - estimate[:, 1:]  # (M, d)
    sigma = np.sqrt(np.diag(kde.covariance))  # (d,)
    moment1 = residual / sigma[None, :]  # (M, d)

    return f, moment1


def _corrected_density(f, m1, a0, a1, A2, order):
    """Compute boundary-corrected density from KDE and truncated moments.

    Implements renormalisation (order 0) and local-linear correction
    (order 1) from the generic vector/matrix moments ``a0``, ``a1``, ``A2``.

    For ``d = 1`` this reduces exactly to the scalar Jones (1993), Eq. (3.4).
    https://doi.org/10.1007/BF00147776

    For ``d = 2`` this follows Chacón & Duong (2018), Sec. 4.3.2, Eq. (4.4).
    https://doi.org/10.1201/9780429485572-4
    (Note, our notation matches Jones, not Chacón & Duong.)

    Parameters
    ----------
    f : np.array
        Uncorrected KDE density, shape ``(M,)``.
    m1 : np.array
        First bandwidth-scaled kernel moment, shape ``(M, d)``.
    a0 : np.array
        Zeroth truncated kernel moment, shape ``(M,)``.
    a1 : np.array
        First bandwidth-scaled truncated kernel moment, shape ``(M, d)``.
    A2 : np.array
        Second bandwidth-scaled truncated kernel moment, shape ``(M, d, d)``.
    order : int
        Boundary correction order (0 or 1).

    Returns
    -------
    p : np.array
        Corrected density, clamped to non-negative values. The linear
        correction can produce negative values near boundaries; when the local
        linear system is singular or ill-conditioned, the correction falls back
        to renormalisation.

    """
    p = np.zeros_like(f)  # (M,)

    # Order-0 correction: renormalize by the truncated kernel mass.
    mask = a0 > 0  # (M,)
    p[mask] = f[mask] / a0[mask]

    if order == 0:
        np.maximum(p, 0, out=p)
        return p
    if order != 1:
        raise ValueError(f"order must be 0 or 1, got {order}")

    # Order-1 correction: solve the local linear system built from a1 and A2.
    determinant = np.linalg.det(A2)  # (M,)
    mask &= np.isfinite(determinant) & (determinant > 0)

    if np.any(mask):
        # Solve A2 x = [m1, a1] in one batched call.
        rhs = np.stack([m1[mask], a1[mask]], axis=-1)  # (M', d, 2)
        sol = np.linalg.solve(A2[mask], rhs)  # (M', d, 2)
        A2m1 = np.zeros_like(m1)  # (M, d)
        A2a1 = np.zeros_like(a1)  # (M, d)
        A2m1[mask] = sol[..., 0]
        A2a1[mask] = sol[..., 1]

        # p = (f - a1^T A2^{-1} m1) / (a0 - a1^T A2^{-1} a1)
        # Reduces to (a2 f - a1 m1) / (a0 a2 - a1^2) in 1D.
        numerator = f - np.einsum('mi,mi->m', a1, A2m1)  # (M,)
        denominator = a0 - np.einsum('mi,mi->m', a1, A2a1)  # (M,)

        # Apply the linear correction only where the system stays well-defined.
        mask &= denominator > 0
        mask &= np.isfinite(numerator) & np.isfinite(denominator)
        p[mask] = numerator[mask] / denominator[mask]

    # Local-linear correction can go negative near boundaries; clamp to zero.
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
        * 0: renormalisation --- O(h) bias.
        * 1: linear correction (Jones 1993, Eq. 3.4) --- O(h²) bias.

    Returns
    -------
    p : np.array
        Boundary-corrected density values.

    """
    if xmin is None and xmax is None or order < 0:
        return kde(x)

    # (2, d)
    x_limits = np.array([[-np.inf if xmin is None else xmin],
                         [np.inf if xmax is None else xmax]], dtype=float)

    # Evaluate the raw KDE, truncated moments, and corrected density.
    f, m1 = _kde_eval(kde, x[:, None])
    a0, a1, A2 = _truncated_moments(x[:, None], kde.covariance, x_limits)
    p = _corrected_density(f, m1, a0, a1, A2, order)

    if xmin is not None:
        p[x < xmin] = 0
    if xmax is not None:
        p[x > xmax] = 0

    return p


def boundary_correction_2d(kde, X, Y, order=1,
                           xmin=None, xmax=None, ymin=None, ymax=None):
    r"""Boundary correction for a 2D Gaussian KDE.

    Applies renormalisation (order 0) or the full non-separable linear
    correction (order 1) using the full kernel covariance matrix.

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
        * 0: renormalisation --- O(h) bias.
        * 1: linear correction --- O(h²) bias.

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

    # (2, d)
    x_limits = np.array([[-np.inf if xmin is None else xmin,
                          -np.inf if ymin is None else ymin],
                         [np.inf if xmax is None else xmax,
                          np.inf if ymax is None else ymax]], dtype=float)

    # Evaluate the raw KDE, truncated moments, and corrected density.
    coords = np.column_stack([x, y])  # (M, d)
    f, m1 = _kde_eval(kde, coords)
    a0, a1, A2 = _truncated_moments(coords, kde.covariance, x_limits)
    p = _corrected_density(f, m1, a0, a1, A2, order)

    if xmin is not None:
        p[x < xmin] = 0
    if xmax is not None:
        p[x > xmax] = 0
    if ymin is not None:
        p[y < ymin] = 0
    if ymax is not None:
        p[y > ymax] = 0

    return p.reshape(X.shape)
