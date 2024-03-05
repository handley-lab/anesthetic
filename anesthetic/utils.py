"""Data-processing utility functions."""
import numpy as np
import pandas
from scipy import special
from scipy.interpolate import interp1d
from scipy.stats import kstwobign, entropy
from matplotlib.tri import Triangulation
import contextlib
import inspect
import re


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    r"""Compute the log of the sum of exponentials of input elements.

    This function has the same call signature as
    :func:`scipy.special.logsumexp` and mirrors scipy's behaviour except for
    ``-np.inf`` input. If a and b are both ``-inf`` then scipy's function will
    output ``nan`` whereas here we use:

    .. math::

        \lim_{x \to -\infty} x \exp(x) = 0

    Thus, if ``a=-inf`` in ``log(sum(b * exp(a))`` then we can set ``b=0``
    such that that term is ignored in the sum.
    """
    if b is None:
        b = np.ones_like(a)
    b = np.where(a == -np.inf, 0, b)
    return special.logsumexp(a, axis=axis, b=b, keepdims=keepdims,
                             return_sign=return_sign)


def neff(w, beta=1):
    r"""Calculate effective number of samples.

    Using the Huggins-Roy family of effective samples
    (https://aakinshin.net/posts/huggins-roy-ess/).

    Parameters
    ----------
    beta : int, float, str, default = 1
        The value of beta used to calculate the number of effective samples
        according to

        .. math::

            N_{eff} &= \bigg(\sum_{i=0}^n w_i^\beta \bigg)^{\frac{1}{1-\beta}}

            w_i &= \frac{w_i}{\sum_j w_j}

        Beta can take any positive value. Larger beta corresponds to a greater
        compression such that:

        .. math::

            \beta_1 < \beta_2 \Rightarrow N_{eff}(\beta_1) > N_{eff}(\beta_2)

        Alternatively, beta can take one of the following strings as input:

        * If 'inf' or 'equal' is supplied (equivalent to beta=inf), then the
          resulting number of samples is the number of samples when compressed
          to equal weights, and given by:

        .. math::

            w_i &= \frac{w_i}{\sum_j w_j}

            N_{eff} &= \frac{1}{\max_i[w_i]}

        * If 'entropy' is supplied (equivalent to beta=1), then the estimate
          is determined via the entropy based calculation, also referred to as
          the channel capacity:

        .. math::

            H &= -\sum_i p_i \ln p_i

            p_i &= \frac{w_i}{\sum_j w_j}

            N_{eff} &= e^{H}

        * If 'kish' is supplied (equivalent to beta=2), then a Kish estimate
          is computed (Kish, Leslie (1965). Survey Sampling.
          New York: John Wiley & Sons, Inc. ISBN 0-471-10949-5):

        .. math::

            N_{eff} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}

        * str(float) input gets converted to the corresponding float value.

    """
    w = w / np.sum(w)
    if beta == np.inf or beta == 'inf' or beta == 'equal':
        return 1 / np.max(w)
    elif beta == 'entropy' or beta != 'kish' and str(float(beta)) == '1.0':
        return np.exp(entropy(w))
    else:
        if beta == 'kish':
            beta = 2
        elif isinstance(beta, str):
            beta = float(beta)
        return np.sum(w**beta)**(1/(1-beta))


def compress_weights(w, u=None, ncompress=True):
    """Compresses weights to their approximate channel capacity."""
    if u is None:
        u = np.random.rand(len(w))

    if w is None:
        w = np.ones_like(u)

    if ncompress is True or isinstance(ncompress, str):
        if ncompress is True:
            ncompress = 'entropy'
        ncompress = neff(w, beta=ncompress)
    elif ncompress is False:
        return w

    # TODO: remove this in version >= 2.1
    if ncompress < 0:
        raise ValueError(
            "ncompress<0 is anesthetic 1.0 behaviour. For equally weighted "
            "samples you should now use ncompress='inf' or ncompress='equal'."
            )

    W = w * ncompress / w.sum()

    fraction, integer = np.modf(W)
    extra = (u < fraction).astype(int)
    return (integer + extra).astype(int)


def quantile(a, q, w=None, interpolation='linear'):
    """Compute the weighted quantile for a one dimensional array."""
    if w is None:
        w = np.ones_like(a)
    a = np.array(list(a))  # Necessary to convert pandas arrays
    w = np.array(list(w))  # Necessary to convert pandas arrays
    i = np.argsort(a)
    c = np.cumsum(w[i[1:]]+w[i[:-1]])
    c = c / c[-1]
    c = np.concatenate(([0.], c))
    icdf = interp1d(c, a[i], kind=interpolation)
    quant = icdf(q)
    if isinstance(q, float):
        quant = float(quant)
    return quant


def mirror_1d(d, xmin=None, xmax=None):
    """If necessary apply reflecting boundary conditions."""
    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        return np.concatenate((2*xmin-d[d < xmed], d, 2*xmax-d[d >= xmed]))
    elif xmin is not None:
        return np.concatenate((2*xmin-d, d))
    elif xmax is not None:
        return np.concatenate((d, 2*xmax-d))
    else:
        return d


def mirror_2d(d_x_, d_y_, xmin=None, xmax=None, ymin=None, ymax=None):
    """If necessary apply reflecting boundary conditions."""
    d_x = d_x_.copy()
    d_y = d_y_.copy()

    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        d_y = np.concatenate((d_y[d_x < xmed], d_y, d_y[d_x >= xmed]))
        d_x = np.concatenate((2*xmin-d_x[d_x < xmed], d_x,
                              2*xmax-d_x[d_x >= xmed]))
    elif xmin is not None:
        d_y = np.concatenate((d_y, d_y))
        d_x = np.concatenate((2*xmin-d_x, d_x))
    elif xmax is not None:
        d_y = np.concatenate((d_y, d_y))
        d_x = np.concatenate((d_x, 2*xmax-d_x))

    if ymin is not None and ymax is not None:
        ymed = (ymin+ymax)/2
        d_x = np.concatenate((d_x[d_y < ymed], d_x, d_x[d_y >= ymed]))
        d_y = np.concatenate((2*ymin-d_y[d_y < ymed], d_y,
                              2*ymax-d_y[d_y >= ymed]))
    elif ymin is not None:
        d_x = np.concatenate((d_x, d_x))
        d_y = np.concatenate((2*ymin-d_y, d_y))
    elif ymax is not None:
        d_x = np.concatenate((d_x, d_x))
        d_y = np.concatenate((d_y, 2*ymax-d_y))

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

    This has the same arguments and keywords as :func:`numpy.histogram`, but is
    normalised to 1.
    """
    hist, bin_edges = np.histogram(a, **kwargs)
    xpath, ypath = np.empty((2, 4*len(hist)))
    ypath[0::4] = ypath[3::4] = 0
    ypath[1::4] = ypath[2::4] = hist
    xpath[0::4] = xpath[1::4] = bin_edges[:-1]
    xpath[2::4] = xpath[3::4] = bin_edges[1:]
    mx = max(ypath)
    if mx:
        ypath /= max(ypath)
    return xpath, ypath


def histogram_bin_edges(samples, weights, bins='fd', range=None, beta='equal'):
    """Compute a good number of bins dynamically from weighted samples.

    Parameters
    ----------
    samples : array_like
        Input data.

    weights : array-like
        Array of sample weights.

    bins : str, default='fd'
        String defining the rule used to automatically compute a good number
        of bins for the weighted samples:

            * 'fd'    : Freedman--Diaconis rule (modified for weighted data)
            * 'scott' : Scott's rule (modified for weighted data)
            * 'sqrt'  : Square root estimator (modified for weighted data)

    range : (float, float), optional
        The lower and upper range of the bins. If not provided, range is simply
        ``(a.min(), a.max())``. Values outside the range are ignored. The first
        element of the range must be less than or equal to the second.

    beta : float, default='equal'
        The value of beta>0 used to calculate the number of effective
        samples via :func:`neff`.

    Returns
    -------
    bin_edges : array of dtype float
        The edges to pass to :func:`numpy.histogram`.

    """
    if weights is None:
        weights = np.ones_like(samples)
    if range is None:
        minimum = np.min(samples)
        maximum = np.max(samples)
        data_range = maximum - minimum
    else:
        minimum = range[0]
        maximum = range[1]
        data_range = maximum - minimum
        data_mask = (samples >= minimum) & (samples <= maximum)
        samples = samples[data_mask]
        weights = weights[data_mask]
    w = weights / np.sum(weights)
    N_eff = neff(w=w, beta=beta)
    if bins == 'fd':  # Freedman--Diaconis rule
        q1, q3 = quantile(samples, [1/4, 3/4], w=w)
        bin_width = 2 * (q3 - q1) * N_eff**(-1/3)
    elif bins == 'scott':  # Scott's rule
        weighted_mean = np.average(samples, weights=w)
        weighted_var = np.average((samples - weighted_mean)**2, weights=w)
        weighted_std = np.sqrt(weighted_var)
        bin_width = (24 * np.pi**0.5 / N_eff)**(1/3) * weighted_std
    elif bins == 'sqrt':  # Square root estimator
        samples_eff, _ = sample_compression_1d(samples, w=w, ncompress=N_eff)
        data_range_eff = np.max(samples_eff) - np.min(samples_eff)
        bin_width = data_range_eff / np.sqrt(N_eff)
    nbins = int(np.ceil(data_range / bin_width))
    bin_edges = np.linspace(minimum, maximum, nbins+1)
    return bin_edges


def compute_nlive(death, birth):
    """Compute number of live points from birth and death contours.

    Parameters
    ----------
    death, birth : array-like
        list of birth and death contours

    Returns
    -------
    nlive : np.array
        number of live points at each contour
    """
    b = pandas.DataFrame(np.array(birth), columns=['logL'])
    d = pandas.DataFrame(np.array(death), columns=['logL'],
                         index=b.index + len(b))
    b['n'] = +1
    d['n'] = -1
    t = pandas.concat([b, d]).sort_values(['logL', 'n'])
    n = t.n.cumsum()
    return (n[d.index]+1).to_numpy()


def compute_insertion_indexes(death, birth):
    """Compute the live point insertion index for each point.

    For more detail, see `Fowlie et al. (2020)
    <https://arxiv.org/abs/2006.03371>`_

    Parameters
    ----------
    death, birth : array-like
        list of birth and death contours

    Returns
    -------
    indexes : np.array
        live point index at which each live point was inserted
    """
    indexes = np.zeros_like(birth, dtype=int)
    for i, (b, d) in enumerate(zip(birth, death)):
        i_live = (death > b) & (birth <= b)
        live = death[i_live]
        live.sort()
        indexes[i] = np.searchsorted(live, d)
    return indexes


def unique(a):
    """Find unique elements, retaining order."""
    b = []
    for x in a:
        if x not in b:
            b.append(x)
    return b


def iso_probability_contours(pdf, contours=[0.95, 0.68]):
    """Compute the iso-probability contour values."""
    if len(contours) > 1 and not np.all(contours[:-1] > contours[1:]):
        raise ValueError(
            "The kwargs `levels` and `contours` have to be ordered from "
            "outermost to innermost contour, i.e. in strictly descending "
            "order when referring to the enclosed probability mass, e.g. "
            "like the default [0.95, 0.68]. "
            "This breaking change in behaviour was introduced in version "
            "2.0.0-beta.10, in order to better match the ordering of other "
            "matplotlib kwargs."
        )
    contours = [1-p for p in contours]
    p = np.sort(np.array(pdf).flatten())
    m = np.cumsum(p)
    m /= m[-1]
    interp = interp1d([0]+list(m), [0]+list(p))
    c = list(interp(contours))+[max(p)]

    return c


def iso_probability_contours_from_samples(pdf, contours=[0.95, 0.68],
                                          weights=None):
    """Compute the iso-probability contour values."""
    if len(contours) > 1 and not np.all(contours[:-1] > contours[1:]):
        raise ValueError(
            "The kwargs `levels` and `contours` have to be ordered from "
            "outermost to innermost contour, i.e. in strictly descending "
            "order when referring to the enclosed probability mass, e.g. "
            "like the default [0.95, 0.68]. "
            "This breaking change in behaviour was introduced in version "
            "2.0.0-beta.10, in order to better match the ordering of other "
            "matplotlib kwargs."
        )
    if weights is None:
        weights = np.ones_like(pdf)
    contours = [1-p for p in contours]
    i = np.argsort(pdf)
    m = np.cumsum(weights[i])
    m /= m[-1]
    interp = interp1d([0]+list(m), [0]+list(pdf[i]))
    c = list(interp(contours))+[max(pdf)]

    return c


def scaled_triangulation(x, y, cov):
    """Triangulation scaled by a covariance matrix.

    Parameters
    ----------
    x, y : array-like
        x and y coordinates of samples

    cov : array-like, 2d
        Covariance matrix for scaling

    Returns
    -------
    :class:`matplotlib.tri.Triangulation`
        Triangulation with the appropriate scaling
    """
    L = np.linalg.cholesky(cov)
    Linv = np.linalg.inv(L)
    x_, y_ = Linv.dot([x, y])
    tri = Triangulation(x_, y_)
    return Triangulation(x, y, tri.triangles)


def triangular_sample_compression_2d(x, y, cov, w=None, n=1000):
    """Histogram a 2D set of weighted samples via triangulation.

    This defines bins via a triangulation of the subsamples and sums weights
    within triangles surrounding each point

    Parameters
    ----------
    x, y : array-like
        x and y coordinates of samples for compressing

    cov : array-like, 2d
        Covariance matrix for scaling

    w : :class:`pandas.Series`, optional
        weights of samples

    n : int, default=1000
        number of samples returned.

    Returns
    -------
    tri :
        :class:`matplotlib.tri.Triangulation` with an appropriate scaling

    w : array-like
        Compressed samples and weights
    """
    # Pre-process samples to not be affected by non-standard indexing
    # Details: https://github.com/handley-lab/anesthetic/issues/189
    x = np.array(x)
    y = np.array(y)

    x = pandas.Series(x)
    if w is None:
        w = pandas.Series(index=x.index, data=np.ones_like(x))

    if n is False:
        n = len(x)
    elif n is True or isinstance(n, str):
        if n is True:
            n = 'entropy'
        n = int(neff(w, beta=n))

    # Select samples for triangulation
    if (w != 0).sum() < n:
        i = x.index
    else:
        i = np.random.choice(x.index, size=n, replace=False, p=w/w.sum())

    # Generate triangulation
    tri = scaled_triangulation(x[i], y[i], cov)

    # For each point find corresponding triangles
    trifinder = tri.get_trifinder()
    j = trifinder(x, y)
    k = tri.triangles[j[j != -1]]

    # Compute mass in each triangle, and add it to each corner
    w_ = np.zeros(len(i))
    for i in range(3):
        np.add.at(w_, k[:, i], w[j != -1]/3)

    return tri, w_


def sample_compression_1d(x, w=None, ncompress=True):
    """Histogram a 1D set of weighted samples via subsampling.

    This compresses the number of samples, combining weights.

    Parameters
    ----------
    x : array-like
        x coordinate of samples for compressing

    w : :class:`pandas.Series`, optional
        weights of samples

    ncompress : int, default=True
        Degree of compression.

        * If ``int``: number of samples returned.
        * If ``True``: compresses to the channel capacity
          (same as ``ncompress='entropy'``).
        * If ``False``: no compression.
        * If ``str``: determine number from the Huggins-Roy family of effective
          samples in :func:`neff` with ``beta=ncompress``.

    Returns
    -------
    x, w: array-like
        Compressed samples and weights
    """
    if ncompress is False:
        return x, w
    elif ncompress is True or isinstance(ncompress, str):
        if ncompress is True:
            ncompress = 'entropy'
        ncompress = neff(w, beta=ncompress)
    x = np.array(x)
    if w is None:
        w = np.ones_like(x)
    w = np.array(w)

    # Select inner samples for triangulation
    if len(x) > ncompress:
        x_ = np.random.choice(x, size=int(ncompress), replace=False)
    else:
        x_ = x.copy()
    x_.sort()

    # Compress mass onto these subsamples
    centers = (x_[1:] + x_[:-1])/2
    j = np.digitize(x, centers)
    w_ = np.zeros_like(x_)
    np.add.at(w_, j, w)

    return x_, w_


def is_int(x):
    """Test whether x is an integer."""
    return isinstance(x, int) or isinstance(x, np.integer)


def match_contour_to_contourf(contours, vmin, vmax):
    """Get needed `vmin, vmax` to match `contour` colors to `contourf` colors.

    `contourf` uses the arithmetic mean of contour levels to assign colors,
    whereas `contour` uses the contour level directly. To get the same colors
    for `contour` lines as for `contourf` faces, we need some fiddly algebra.
    """
    if len(contours) <= 2:
        vmin = 2 * vmin - vmax
        return vmin, vmax
    else:
        c0 = contours[0]
        c1 = contours[1]
        ce = contours[-2]
        denom = vmax + ce - c1 - c0
        vmin = +(c0 * vmax - c1 * ce + 2 * vmin * (ce - c0)) / denom
        vmax = -(c0 * vmax + c1 * ce - 2 * vmax * ce) / denom
        return vmin, vmax


def insertion_p_value(indexes, nlive, batch=0):
    """Compute the p-value from insertion indexes, assuming constant nlive.

    Note that this function doesn't use :func:`scipy.stats.kstest` as the
    latter assumes continuous distributions.

    For more detail, see `Fowlie et al. (2020)
    <https://arxiv.org/abs/2006.03371>`_

    For a rolling test, you should provide the optional parameter ``batch!=0``.
    In this case the test computes the p-value on consecutive batches of size
    ``nlive * batch``, selects the smallest one and adjusts for multiple
    comparisons using a Bonferroni correction.

    Parameters
    ----------
    indexes : array-like
        list of insertion indexes, sorted by death contour

    nlive : int
        number of live points

    batch : float
        batch size in units of nlive for a rolling p-value

    Returns
    -------
    ks_result : dict
        Kolmogorov-Smirnov test results:

            * ``D``: Kolmogorov-Smirnov statistic
            * ``sample_size``: sample size
            * ``p-value``: p-value

            if ``batch != 0``:

            * ``iterations``: bounds of batch with minimum p-value
            * ``nbatches``: the number of batches in total
            * ``uncorrected p-value``: p-value without Bonferroni correction
    """
    if batch == 0:
        bins = np.arange(-0.5, nlive + 0.5, 1.)
        empirical_pmf = np.histogram(
            np.array(indexes),
            bins=bins,
            density=True,
        )[0]
        empirical_cmf = np.cumsum(empirical_pmf)
        uniform_cmf = np.arange(1., nlive + 1., 1.) / nlive

        D = abs(empirical_cmf - uniform_cmf).max()
        sample_size = len(indexes)
        K = D * np.sqrt(sample_size)

        ks_result = {}
        ks_result["D"] = D
        ks_result["sample_size"] = sample_size
        ks_result["p-value"] = kstwobign.sf(K)
        return ks_result
    else:
        batch = int(batch * nlive)
        batches = [
            np.array(indexes)[i:i + batch]
            for i in range(0, len(indexes), batch)
        ]
        ks_results = [insertion_p_value(c, nlive) for c in batches]
        ks_result = min(ks_results, key=lambda t: t["p-value"])
        index = ks_results.index(ks_result)

        ks_result["iterations"] = (index * batch, (index + 1) * batch)
        ks_result["nbatches"] = n = len(batches)
        ks_result["uncorrected p-value"] = p = ks_result["p-value"]
        ks_result["p-value"] = 1. - (1. - p)**n if p != 1 else p * n
        return ks_result


@contextlib.contextmanager
def temporary_seed(seed):
    """Context for temporarily setting a numpy seed."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def adjust_docstrings(obj, pattern, repl, *args, **kwargs):
    """Adjust the docstrings of a class using regular expressions.

    After the first argument, the remaining arguments are identical to re.sub.

    Parameters
    ----------
    cls : class
        class to adjust

    pattern : str
        regular expression pattern

    repl : str
        replacement string
    """
    if inspect.isclass(obj):
        for key, val in obj.__dict__.items():
            doc = inspect.getdoc(val)
            if doc is not None:
                newdoc = re.sub(pattern, repl, doc, *args, **kwargs)
                try:
                    obj.__dict__[key].__doc__ = newdoc
                except AttributeError:
                    pass
    else:
        doc = inspect.getdoc(obj)
        if doc is not None:
            newdoc = re.sub(pattern, repl, doc, *args, **kwargs)
            obj.__doc__ = newdoc
