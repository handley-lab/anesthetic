"""Data-processing utility functions."""
import numpy
import pandas
from scipy.interpolate import interp1d


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

    if nsamples is not None:
        W = w/w.max()
        if nsamples > 0 and sum(W) > nsamples:
            W = W/sum(W)*nsamples
    else:
        W = w * channel_capacity(w) / w.sum()

    fraction, integer = numpy.modf(W)
    extra = (u < fraction).astype(int)
    return (integer + extra).astype(int)


def quantile(a, q, w):
    """Compute the weighted quantile for a one dimensional array."""
    if w is None:
        w = numpy.ones_like(a)
    i = numpy.argsort(a)
    c = numpy.cumsum(w[i])
    c /= c[-1]
    icdf = interp1d(c/c[-1], a[i])
    return icdf(q)


def check_bounds(d, xmin=None, xmax=None):
    """Check if we need to apply strict bounds."""
    if xmin is not None and (d.min() - xmin)/(d.max()-d.min()) > 1e-2:
        xmin = None
    if xmax is not None and (xmax - d.max())/(d.max()-d.min()) > 1e-2:
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
    """Compute number of live points from birth and death contours."""
    contours = numpy.concatenate(([birth[0]], death))
    index = numpy.arange(death.size)
    birth_index = contours.searchsorted(birth)-1
    births = pandas.Series(+1, index=birth_index).sort_index()
    deaths = pandas.Series(-1, index=index)
    nlive = pandas.concat([births, deaths]).sort_index()
    nlive = nlive.groupby(nlive.index).sum().cumsum()
    return nlive.values[:-1]


def unique(a):
    """Find unique elements, retaining order."""
    b = []
    for x in a:
        if x not in b:
            b.append(x)
    return b
