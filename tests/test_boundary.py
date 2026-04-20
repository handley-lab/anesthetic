import pytest
from numpy.ma.testutils import assert_array_equal

import anesthetic.examples._matplotlib_agg  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from anesthetic.plot import kde_plot_1d, kde_contour_plot_2d
from anesthetic.boundary import (boundary_correction_2d, _truncated_moments,
                                 _bvn_cdf)


def test_boundary_errors():
    # _bvn_cdf: all-nonfinite early-return path.
    x = np.array([np.inf, -np.inf])
    y = np.array([-np.inf, np.inf])
    cdf = _bvn_cdf(x, y, rho=0.5)
    assert_array_equal(cdf, [0, 0])

    # _truncated_moments: unsupported dimensionality (d != 1, 2).
    x = np.random.randn(10, 3)
    cov = np.eye(3)
    x_limits = np.array([[-1, -1, -1], [1, 1, 1]])
    with pytest.raises(NotImplementedError, match="only 1D and 2D"):
        _truncated_moments(x, cov, x_limits)

    # _truncated_moments: cov shape mismatch.
    x = np.random.randn(10, 2)
    cov = np.eye(3)
    x_limits = np.array([[-1, -1], [1, 1]])
    with pytest.raises(ValueError, match="cov must have shape"):
        _truncated_moments(x, cov, x_limits)

    # _truncated_moments: x_limits shape mismatch.
    x = np.random.randn(10, 2)
    cov = np.eye(2)
    x_limits = np.array([[-1, -1, -1], [1, 1, 1]])
    with pytest.raises(ValueError, match="x_limits must have shape"):
        _truncated_moments(x, cov, x_limits)


def test_boundary_correction_1d():
    """Order 1 correction is closer to the truth than order 0."""
    np.random.seed(42)
    d = np.random.uniform(low=-1, high=2, size=10000)
    w = stats.norm.pdf(d)
    x = np.linspace(d.min(), d.max(), 301)
    truth = stats.norm.pdf(x) / stats.norm.pdf(x).max()

    _, ax = plt.subplots()
    t, = ax.plot(x, truth)
    kwargs = dict(bw_method=0.25, q=0, nplot_1d=x.size)
    pn, = kde_plot_1d(ax, d, weights=w, order=-1, **kwargs)  # no correction
    p0, = kde_plot_1d(ax, d, weights=w, order=+0, **kwargs)  # order 0
    p1, = kde_plot_1d(ax, d, weights=w, order=+1, **kwargs)  # order 1
    with pytest.raises(ValueError):
        kde_plot_1d(ax, d, weights=w, order=2, **kwargs)  # order 2 n.a.

    assert np.all(pn.get_ydata() >= 0)
    assert np.all(p0.get_ydata() >= 0)
    assert np.all(p1.get_ydata() >= 0)

    assert_array_equal(pn.get_xdata(), t.get_xdata())
    assert_array_equal(p0.get_xdata(), t.get_xdata())
    assert_array_equal(p1.get_xdata(), t.get_xdata())

    residual_n = np.abs(pn.get_ydata() / t.get_ydata() - 1)
    residual_0 = np.abs(p0.get_ydata() / t.get_ydata() - 1)
    residual_1 = np.abs(p1.get_ydata() / t.get_ydata() - 1)

    assert residual_1.max() < residual_0.max() < residual_n.max() < 0.5
    assert residual_1.mean() < residual_0.mean() < residual_n.mean() < 0.1


def test_boundary_correction_2d():
    """Order 1 correction is closer to the truth than order 0 in 2D."""
    np.random.seed(42)
    lo, hi = -1, 2
    d = np.random.uniform(lo, hi, (10000, 2))
    w = stats.multivariate_normal(mean=[0, 0]).pdf(d)

    kde = gaussian_kde(d.T, weights=w, bw_method=0.4)

    X, Y = np.mgrid[lo:hi:20j, lo:hi:20j]
    truth = stats.multivariate_normal(mean=[0, 0]).pdf(np.dstack([X, Y]))
    truth /= truth.max()

    bounds = dict(xmin=lo, xmax=hi, ymin=lo, ymax=hi)
    pn = boundary_correction_2d(kde, X, Y, order=-1, **bounds)
    p0 = boundary_correction_2d(kde, X, Y, order=+0, **bounds)
    p1 = boundary_correction_2d(kde, X, Y, order=+1, **bounds)
    pn /= pn.max()
    p0 /= p0.max()
    p1 /= p1.max()
    assert np.all(pn >= 0)
    assert np.all(p0 >= 0)
    assert np.all(p1 >= 0)

    residual_n = np.abs(pn - truth)
    residual_0 = np.abs(p0 - truth)
    residual_1 = np.abs(p1 - truth)

    assert residual_1.max() < residual_n.max()
    assert residual_1.max() < residual_0.max()
    assert residual_1.mean() < residual_n.mean()
    assert residual_1.mean() < residual_0.mean()


def test_full_covariance_2d():
    """Full-covariance truncated moments differ from separable ones.

    With a correlated kernel (rho=0.8) near a corner, the truncated moments
    from the full-covariance computation should differ from those obtained
    with a diagonal covariance (same marginal variances but rho=0). This is
    the distinguishing feature of the full vs separable boundary correction.
    """
    np.random.seed(42)
    rho = 0.8
    cov_full = np.array([[1, rho],
                         [rho, 1]])
    cov_diag = np.diag(np.diag(cov_full))

    lower = 0
    upper = 1
    x_limits = np.array([[lower, lower],
                         [upper, upper]])

    # Evaluate near the corner where correlation matters most.
    x = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    a0_f, a1_f, A2_f = _truncated_moments(x, cov_full, x_limits)
    a0_d, a1_d, A2_d = _truncated_moments(x, cov_diag, x_limits)

    # All moments must differ between correlated and uncorrelated.
    assert not np.allclose(a0_f, a0_d, atol=1e-4), \
        "a0 should differ between full and diagonal covariance"
    assert not np.allclose(a1_f, a1_d, atol=1e-4), \
        "a1 should differ between full and diagonal covariance"
    assert not np.allclose(A2_f, A2_d, atol=1e-4), \
        "A2 should differ between full and diagonal covariance"

    # Also check a grid of points to verify the correction differs
    # at a practical level through the full pipeline.
    d = np.random.multivariate_normal([1, 1], cov_full, size=5000)
    d = d[(d[:, 0] > lower) & (d[:, 1] > lower)]
    d = d[(d[:, 0] < upper) & (d[:, 1] < upper)]
    kde = gaussian_kde(d.T)

    X, Y = np.mgrid[lower:upper:20j, lower:upper:20j]
    bounds = dict(xmin=lower, xmax=upper, ymin=lower, ymax=upper)
    p = boundary_correction_2d(kde, X, Y, order=1, **bounds)

    # Sanity: corrected density is non-negative and non-trivial.
    assert np.all(p >= 0)
    assert p.max() > 0


def test_bw_scale_1d():
    "Larger bw_scale means more smoothing which flattens the peak."
    np.random.seed(43)
    d = np.random.standard_normal(1000)
    _, ax = plt.subplots()
    kwargs = dict(q=0, density=True)
    narrow, = kde_plot_1d(ax, d, bw_scale=0.5, **kwargs)
    default, = kde_plot_1d(ax, d, bw_scale=1.0, **kwargs)
    wide, = kde_plot_1d(ax, d, bw_scale=1.5, **kwargs)

    assert narrow.get_ydata().max() > default.get_ydata().max()
    assert default.get_ydata().max() > wide.get_ydata().max()


def test_bw_scale_2d():
    "Larger bw_scale means more smoothing which flattens the peak in 2D."
    np.random.seed(43)
    d = np.random.standard_normal((1000, 2))
    kwargs = dict(q=0, facecolor=True)
    peaks = []
    for scale in [0.5, 1.0, 2.0]:
        _, ax = plt.subplots()
        contf, _ = kde_contour_plot_2d(ax, d[:, 0], d[:, 1],
                                       bw_scale=scale, **kwargs)
        peaks.append(contf.norm.vmax)

    assert peaks[0] > peaks[1] > peaks[2]
