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
from scipy.special import ndtr
from scipy.stats import norm
from scipy.stats._kde import _get_output_dtype
from scipy.stats._stats import gaussian_kernel_estimate

# Pre-compute 20-point Gauss-Legendre nodes and weights for _bvn_cdf.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(20)


def _bvn_cdf(x, y, rho):
    r"""Vectorized bivariate standard-normal CDF.

    Computes `P(X < x, Y < y)` where `(X, Y)` follow a standard bivariate
    normal distribution with correlation `rho`, using the Drezner (1978)
    integral representation with 20-point Gauss-Legendre quadrature.

    Parameters
    ----------
    x, y : np.array
        Evaluation coordinates (arbitrary but matching shape).
    rho : float
        Correlation coefficient, ``-1 < rho < 1``.

    Returns
    -------
    cdf : np.array
        Bivariate normal CDF values, same shape as `x` and `y`.

    """
    cdf = np.empty_like(x)

    # infinities block
    fin = np.isfinite(x) & np.isfinite(y)
    cdf[~fin] = ndtr(x[~fin]) * ndtr(y[~fin])
    if not np.any(fin):
        return cdf

    # finite block
    xf = x[fin]
    yf = y[fin]
    theta = np.arcsin(rho)
    t = theta / 2 * (_GL_NODES + 1)
    w = theta / 2 * _GL_WEIGHTS
    sin_t = np.sin(t)
    cos2_t = np.cos(t)**2
    X = xf[:, None]
    Y = yf[:, None]
    arg = np.exp(-(X**2 + Y**2 - 2 * X * Y * sin_t) / (2 * cos2_t))
    cdf[fin] = (ndtr(xf) * ndtr(yf) + np.sum(arg * w, axis=-1) / (2 * np.pi))
    return cdf


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
    gaussian = norm(loc=0, scale=1)

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
    if d == 1:
        cdf = gaussian.cdf(u_corner[..., 0])  # (M, 2)
    else:
        rho = np.clip(corr[0, 1], -1 + eps, 1 - eps)
        s = np.sqrt(1 - rho**2)
        cdf = _bvn_cdf(u_corner[..., 0], u_corner[..., 1], rho)  # (M, 2, 2)
    pdf = gaussian.pdf(u_corner)  # (M, *(2,)*d, d)

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
            uj = u_corner[..., j]
            v = (uj - rho * ui) / s
            v = np.where(np.isposinf(uj), +np.inf, v)
            v = np.where(np.isneginf(uj), -np.inf, v)
            inf_i_fin_j = inf_i & np.isfinite(uj)
            if rho == 0:
                v = np.where(inf_i_fin_j, uj / s, v)
            else:
                v = np.where(inf_i_fin_j, -np.sign(rho) * ui, v)
            grad[..., i] *= gaussian.cdf(v)
            hess[..., i, j] = pdf[..., i] * gaussian.pdf(v) / s
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
    output_dtype, spec = _get_output_dtype(kde.covariance, x)

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


def boundary_correction_2d(kde, X, Y, order=None,
                           xmin=None, xmax=None, ymin=None, ymax=None,
                           n_vec=None, nmin=None, nmax=None):
    r"""Boundary correction for a 2D Gaussian KDE.

    Applies renormalisation (order 0) or the full non-separable linear
    correction (order 1) using the full kernel covariance matrix on the
    axis-aligned ``[xmin, xmax] x [ymin, ymax]`` rectangle. Optionally
    multiplies in a separable Jones-style 1D correction for an additional
    half-plane bounded by ``nmin <= n_vec . (x, y) <= nmax``.

    Parameters
    ----------
    kde : scipy.stats.gaussian_kde
        Fitted 2D KDE object.
    X, Y : np.array
        2D evaluation grids (from np.mgrid).
    xmin, xmax, ymin, ymax : float, optional
        Axis-aligned bounds.
    order : int
        Boundary correction order.

        * < 0: no correction --- return raw KDE estimate.
        * 0: renormalisation --- O(h) bias.
        * 1: linear correction --- O(h²) bias.

        Default: ``order=1`` for x, y correction, ``order=0`` for n correction.

    n_vec : np.array, shape (2,), optional
        Direction vector ``[n_x, n_y]`` defining a rotated coordinate
        ``n = n_vec . (x, y)``. Combined with ``nmin``/``nmax`` to add a
        separable 1D correction along that direction. Skipped if absent
        or already covered by an active ``x`` / ``y`` bound when
        ``n_vec`` is axis-aligned.
    nmin, nmax : float, optional
        Lower / upper bounds on ``n_vec . (x, y)``.

    Returns
    -------
    p : np.array
        Boundary-corrected 2D density, same shape as X and Y.

    """
    x = X.ravel()
    y = Y.ravel()
    order_n = 0 if order is None else order
    order = 1 if order is None else order

    has_x_bounds = xmin is not None or xmax is not None
    has_y_bounds = ymin is not None or ymax is not None
    has_n_bounds = n_vec is not None and (nmin is not None or nmax is not None)

    # Skip the n-correction when its direction duplicates an active x/y bound.
    if has_n_bounds:
        norm = np.linalg.norm(n_vec)
        n_vec = n_vec / norm
        nmin = None if nmin is None else nmin / norm
        nmax = None if nmax is None else nmax / norm
        if np.allclose(np.abs(n_vec), [1, 0], atol=1e-10) and has_x_bounds:
            has_n_bounds = False
        elif np.allclose(np.abs(n_vec), [0, 1], atol=1e-10) and has_y_bounds:
            has_n_bounds = False

    if (not has_x_bounds and not has_y_bounds and not has_n_bounds
            or order < 0):
        return kde([x, y]).reshape(X.shape)

    # Evaluate the raw KDE and first moment once; reused by all corrections.
    coords = np.column_stack([x, y])  # (M, d)
    f, m1 = _kde_eval(kde, coords)

    # Axis-aligned x/y correction over the rectangle.
    if has_x_bounds or has_y_bounds:
        xy_limits = np.array([[-np.inf if xmin is None else xmin,
                               -np.inf if ymin is None else ymin],
                              [+np.inf if xmax is None else xmax,
                               +np.inf if ymax is None else ymax]])
        a0, a1, A2 = _truncated_moments(coords, kde.covariance, xy_limits)
        p = _corrected_density(f, m1, a0, a1, A2, order)
    else:
        p = f.copy()

    # Separable 1D correction along n_vec (Jones 1993, Eq. 3.4) reusing f, m1.
    if has_n_bounds:
        sigma_xy = np.sqrt(np.diag(kde.covariance))  # (d,)
        sigma_n = np.sqrt(n_vec @ kde.covariance @ n_vec)
        m1_n = (m1 * sigma_xy) @ n_vec / sigma_n  # (M,)
        coords_n = coords @ n_vec  # (M,)
        # Projection can put algebraic boundary points one ulp outside.
        for edge in [nmin, nmax]:
            if edge is not None:
                atol = 8 * np.finfo(coords_n.dtype).eps * max(1, abs(edge))
                coords_n[np.isclose(coords_n, edge, rtol=0, atol=atol)] = edge
        n_limits = np.array([[-np.inf if nmin is None else nmin],
                             [+np.inf if nmax is None else nmax]])
        a0_n, a1_n, A2_n = _truncated_moments(coords_n[:, None],
                                              np.array([[sigma_n**2]]),
                                              n_limits)
        p_n = _corrected_density(f, m1_n[:, None], a0_n, a1_n, A2_n, order_n)

        # Combine separably: p <- p_xy * p_n / f. Guard against f == 0.
        ratio = np.zeros_like(f)
        valid = (f > 0) & np.isfinite(f) & np.isfinite(p_n)
        np.divide(p_n, f, out=ratio, where=valid)
        p = np.where(valid, p * ratio, 0.0)
        np.maximum(p, 0, out=p)

        if nmin is not None:
            p[coords_n < nmin] = 0
        if nmax is not None:
            p[coords_n > nmax] = 0

    if xmin is not None:
        p[x < xmin] = 0
    if xmax is not None:
        p[x > xmax] = 0
    if ymin is not None:
        p[y < ymin] = 0
    if ymax is not None:
        p[y > ymax] = 0

    return p.reshape(X.shape)
