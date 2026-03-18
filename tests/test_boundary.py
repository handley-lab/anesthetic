import pytest
from numpy.ma.testutils import assert_array_equal

import anesthetic.examples._matplotlib_agg  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from anesthetic.plot import kde_plot_1d
from anesthetic.boundary import boundary_correction_2d


def test_boundary_correction_1d():
    """Order 1 correction is closer to the truth than order 0."""
    np.random.seed(42)
    d = np.random.uniform(low=-1, high=2, size=10000)
    w = stats.norm.pdf(d)
    x = np.linspace(d.min(), d.max(), 301)
    truth = stats.norm.pdf(x) / stats.norm.pdf(x).max()

    fig, ax = plt.subplots()
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
    pn = boundary_correction_2d(kde, X, Y, kde.covariance, order=-1, **bounds)
    p0 = boundary_correction_2d(kde, X, Y, kde.covariance, order=+0, **bounds)
    p1 = boundary_correction_2d(kde, X, Y, kde.covariance, order=+1, **bounds)
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
