"""Data-processing utility functions."""
import numpy
import pandas
from scipy import special
from scipy.interpolate import interp1d
from matplotlib.tri import Triangulation


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    r"""Compute the log of the sum of exponentials of input elements.

    This function has the same call signature as `scipy.special.logsumexp`
    and mirrors scipy's behaviour except for `-numpy.inf` input. If a and b
    are both -inf then scipy's function will output `nan` whereas here we use:

    .. math::

        \lim_{x \to -\infty} x \exp(x) = 0

    Thus, if a=-inf in `log(sum(b * exp(a))` then we can set b=0 such that
    that term is ignored in the sum.
    """
    if b is None:
        b = numpy.ones_like(a)
    b = numpy.where(a == -numpy.inf, 0, b)
    return special.logsumexp(a, axis=axis, b=b, keepdims=keepdims,
                             return_sign=return_sign)


def channel_capacity(w):
    r"""Channel capacity (effective sample size).

    .. math::

        H = \sum_i p_i \log p_i

        p_i = \frac{w_i}{\sum_j w_j}

        N = e^{-H}
    """
    with numpy.errstate(divide='ignore', invalid='ignore'):
        W = numpy.array(w)/sum(w)
        H = numpy.nansum(numpy.log(W)*W)
        return numpy.exp(-H)


def compress_weights(w, u=None, nsamples=None):
    """Compresses weights to their approximate channel capacity."""
    if u is None:
        u = numpy.random.rand(len(w))

    if w is None:
        w = numpy.ones_like(u)

    if nsamples is None:
        nsamples = channel_capacity(w)

    if nsamples <= 0:
        W = w/w.max()
    else:
        W = w * nsamples / w.sum()

    fraction, integer = numpy.modf(W)
    extra = (u < fraction).astype(int)
    return (integer + extra).astype(int)


def quantile(a, q, w=None):
    """Compute the weighted quantile for a one dimensional array."""
    if w is None:
        w = numpy.ones_like(a)
    a = numpy.array(list(a))  # Necessary to convert pandas arrays
    w = numpy.array(list(w))  # Necessary to convert pandas arrays
    i = numpy.argsort(a)
    c = numpy.cumsum(w[i[1:]]+w[i[:-1]])
    c /= c[-1]
    c = numpy.concatenate(([0.], c))
    icdf = interp1d(c, a[i])
    quant = icdf(q)
    if isinstance(q, float):
        quant = float(quant)
    return quant


def check_bounds(d, xmin=None, xmax=None):
    """Check if we need to apply strict bounds."""
    if len(d) > 0:
        if xmin is not None and (d.min() - xmin) > 1e-2*(d.max()-d.min()):
            xmin = None
        if xmax is not None and (xmax - d.max()) > 1e-2*(d.max()-d.min()):
            xmax = None
    return xmin, xmax


def mirror_1d(d, xmin=None, xmax=None):
    """If necessary apply reflecting boundary conditions."""
    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        return numpy.concatenate((2*xmin-d[d < xmed], d, 2*xmax-d[d >= xmed]))
    elif xmin is not None:
        return numpy.concatenate((2*xmin-d, d))
    elif xmax is not None:
        return numpy.concatenate((d, 2*xmax-d))
    else:
        return d


def mirror_2d(d_x_, d_y_, xmin=None, xmax=None, ymin=None, ymax=None):
    """If necessary apply reflecting boundary conditions."""
    d_x = d_x_.copy()
    d_y = d_y_.copy()

    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        d_y = numpy.concatenate((d_y[d_x < xmed], d_y, d_y[d_x >= xmed]))
        d_x = numpy.concatenate((2*xmin-d_x[d_x < xmed], d_x,
                                 2*xmax-d_x[d_x >= xmed]))
    elif xmin is not None:
        d_y = numpy.concatenate((d_y, d_y))
        d_x = numpy.concatenate((2*xmin-d_x, d_x))
    elif xmax is not None:
        d_y = numpy.concatenate((d_y, d_y))
        d_x = numpy.concatenate((d_x, 2*xmax-d_x))

    if ymin is not None and ymax is not None:
        ymed = (ymin+ymax)/2
        d_x = numpy.concatenate((d_x[d_y < ymed], d_x, d_x[d_y >= ymed]))
        d_y = numpy.concatenate((2*ymin-d_y[d_y < ymed], d_y,
                                 2*ymax-d_y[d_y >= ymed]))
    elif ymin is not None:
        d_x = numpy.concatenate((d_x, d_x))
        d_y = numpy.concatenate((2*ymin-d_y, d_y))
    elif ymax is not None:
        d_x = numpy.concatenate((d_x, d_x))
        d_y = numpy.concatenate((d_y, 2*ymax-d_y))

    return d_x, d_y


def nest_level(lst):
    """Calculate the nesting level of a list."""
    if not isinstance(lst, list):
        return 0
    if not lst:
        return 1
    return max(nest_level(item) for item in lst) + 1


def histogram(a, **kwargs):
    """Produce a histogram for path-based plotting.

    This is a cheap histogram. Necessary if one wants to update the histogram
    dynamically, and redrawing and filling is very expensive.

    This has the same arguments and keywords as numpy.histogram, but is
    normalised to 1.
    """
    hist, bin_edges = numpy.histogram(a, **kwargs)
    xpath, ypath = numpy.empty((2, 4*len(hist)))
    ypath[0::4] = ypath[3::4] = 0
    ypath[1::4] = ypath[2::4] = hist
    xpath[0::4] = xpath[1::4] = bin_edges[:-1]
    xpath[2::4] = xpath[3::4] = bin_edges[1:]
    mx = max(ypath)
    if mx:
        ypath /= max(ypath)
    return xpath, ypath


def compute_nlive(death, birth):
    """Compute number of live points from birth and death contours.

    Parameters
    ----------
    death, birth : array-like
        list of birth and death contours

    Returns
    -------
    nlive: numpy.array
        number of live points at each contour
    """
    birth_index = death.searchsorted(birth)
    births = pandas.Series(+1, index=birth_index).sort_index()
    index = numpy.arange(death.size)
    deaths = pandas.Series(-1, index=index)
    nlive = pandas.concat([births, deaths]).sort_index()
    nlive = nlive.groupby(nlive.index).sum().cumsum()
    return nlive.values


def unique(a):
    """Find unique elements, retaining order."""
    b = []
    for x in a:
        if x not in b:
            b.append(x)
    return b


def iso_probability_contours(pdf, contours=[0.68, 0.95]):
    """Compute the iso-probability contour values."""
    contours = [1-p for p in reversed(contours)]
    p = numpy.sort(numpy.array(pdf).flatten())
    m = numpy.cumsum(p)
    m /= m[-1]
    interp = interp1d([0]+list(m), [0]+list(p))
    c = list(interp(contours))+[max(p)]

    # Correct level sets
    for i in range(1, len(c)):
        if c[i-1] == c[i]:
            for j in range(i):
                c[j] = c[j] - 1e-5

    return c


def iso_probability_contours_from_samples(pdf, contours=[0.68, 0.95],
                                          weights=None):
    """Compute the iso-probability contour values."""
    if weights is None:
        weights = numpy.ones_like(pdf)
    contours = [1-p for p in reversed(contours)]
    i = numpy.argsort(pdf)
    m = numpy.cumsum(weights[i])
    m /= m[-1]
    interp = interp1d([0]+list(m), [0]+list(pdf[i]))
    c = list(interp(contours))+[max(pdf)]

    # Correct level sets
    for i in range(1, len(c)):
        if c[i-1] == c[i]:
            for j in range(i):
                c[j] = c[j] - 1e-5

    return c


def scaled_triangulation(x, y, cov):
    """Triangulation scaled by a covariance matrix.

    Parameters
    ----------
    x, y: array-like
        x and y coordinates of samples

    cov: array-like, 2d
        Covariance matrix for scaling

    Returns
    -------
    matplotlib.tri.Triangulation
        Triangulation with the appropriate scaling
    """
    L = numpy.linalg.cholesky(cov)
    Linv = numpy.linalg.inv(L)
    x_, y_ = Linv.dot([x, y])
    tri = Triangulation(x_, y_)
    return Triangulation(x, y, tri.triangles)


def triangular_sample_compression_2d(x, y, cov, w=None, n=1000):
    """Histogram a 2D set of weighted samples via triangulation.

    This defines bins via a triangulation of the subsamples and sums weights
    within triangles surrounding each point

    Parameters
    ----------
    x, y: array-like
        x and y coordinates of samples for compressing

    cov: array-like, 2d
        Covariance matrix for scaling

    w: pandas.Series, optional
        weights of samples

    n: int, optional
        number of samples returned. Default 1000

    Returns
    -------
    tri:
        matplotlib.tri.Triangulation with an appropriate scaling

    w: array-like
        Compressed samples and weights
    """
    x = pandas.Series(x)
    if w is None:
        w = pandas.Series(index=x.index, data=numpy.ones_like(x))

    # Select samples for triangulation
    if (w != 0).sum() < n:
        i = x.index
    else:
        i = numpy.random.choice(x.index, size=n, replace=False, p=w/w.sum())

    # Generate triangulation
    tri = scaled_triangulation(x[i], y[i], cov)

    # For each point find corresponding triangles
    trifinder = tri.get_trifinder()
    j = trifinder(x, y)
    k = tri.triangles[j[j != -1]]

    # Compute mass in each triangle, and add it to each corner
    w_ = numpy.zeros(len(i))
    for i in range(3):
        numpy.add.at(w_, k[:, i], w[j != -1]/3)

    return tri, w_


def sample_compression_1d(x, w=None, n=1000):
    """Histogram a 1D set of weighted samples via subsampling.

    This compresses the number of samples, combining weights.

    Parameters
    ----------
    x: array-like
        x coordinate of samples for compressing

    w: pandas.Series, optional
        weights of samples

    n: int, optional
        number of samples returned. Default 1000

    Returns
    -------
    x, w, array-like
        Compressed samples and weights
    """
    x = pandas.Series(x)
    if w is None:
        w = pandas.Series(index=x.index, data=numpy.ones_like(x))

    # Select inner samples for triangulation
    if sum(w != 0) < n:
        i = w.index
    else:
        i = numpy.random.choice(w.index, size=n, replace=False, p=w/w.sum())

    # Define sub-samples
    x_ = numpy.sort(x[i])

    # Compress mass onto these subsamples
    j1 = numpy.digitize(x, x_) - 1
    k1 = (j1 > -1) & (j1 < n)
    j2 = numpy.digitize(x, x_, right=True) - 1
    k2 = (j2 > -1) & (j2 < n)

    w_ = numpy.zeros_like(x_)
    numpy.add.at(w_, j1[k1], w[k1])
    numpy.add.at(w_, j2[k2], w[k2])

    return x_, w_


def is_int(x):
    """Test whether x is an integer."""
    return isinstance(x, int) or isinstance(x, numpy.integer)
