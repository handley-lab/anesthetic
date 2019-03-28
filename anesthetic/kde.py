import numpy
import warnings
from fastkde import fastKDE
from anesthetic.utils import check_bounds, mirror_1d, mirror_2d


def kde_1d(d, xmin=None, xmax=None):
    xmin, xmax = check_bounds(d, xmin, xmax)
    axisExpansionFactor = xmax is None or xmin is None
    d_ = mirror_1d(d, xmin, xmax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, x = fastKDE.pdf(d_,axisExpansionFactor=axisExpansionFactor,
                           numPointsPerSigma=10*(2-axisExpansionFactor))
    p *= 2-axisExpansionFactor

    if xmin is not None:
        p = p[x>=xmin]
        x = x[x>=xmin]

    if xmax is not None:
        p = p[x<=xmax]
        x = x[x<=xmax]

    return x, p


def kde_2d(d_x, d_y, xmin=None, xmax=None, ymin=None, ymax=None):
    xmin, xmax = check_bounds(d_x, xmin, xmax)
    ymin, ymax = check_bounds(d_y, ymin, ymax)
    axisExpansionFactor=[xmax is None or xmin is None,
                         ymax is None or ymin is None] 
    d_x_, d_y_ = mirror_2d(d_x, d_y, xmin, xmax, ymin, ymax)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, (x, y) = fastKDE.pdf(d_x_, d_y_, axisExpansionFactor=axisExpansionFactor,
                                numPointsPerSigma=10*(2-axisExpansionFactor[0])*(2-axisExpansionFactor[1]))

    p *= (2-axisExpansionFactor[0])
    p *= (2-axisExpansionFactor[1])
    if xmin is not None:
        p = p[:,x>=xmin]
        x = x[x>=xmin]

    if xmax is not None:
        p = p[:,x<=xmax]
        x = x[x<=xmax]

    if ymin is not None:
        p = p[y>=ymin,:]
        y = y[y>=ymin]

    if ymax is not None:
        p = p[y<=ymax,:]
        y = y[y<=ymax]

    return x, y, p
