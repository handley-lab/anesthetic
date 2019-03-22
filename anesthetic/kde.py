import numpy
import warnings
from fastkde import fastKDE

def mirror_1d(d, xmin=None, xmax=None):
    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        return numpy.concatenate((2*xmin-d[d<xmed],d,2*xmax-d[d>=xmed]))
    elif xmin is not None:
        return numpy.concatenate((2*xmin-d,d))
    elif xmax is not None:
        return numpy.concatenate((d,2*xmax-d))
    else:
        return d


def mirror_2d(d_x_, d_y_, xmin=None, xmax=None, ymin=None, ymax=None):
    d_x = d_x_.copy()
    d_y = d_y_.copy()

    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        d_y = numpy.concatenate((d_y[d_x<xmed],d_y,d_y[d_x>=xmed]))
        d_x = numpy.concatenate((2*xmin-d_x[d_x<xmed],d_x,2*xmax-d_x[d_x>=xmed]))
    elif xmin is not None:
        d_y = numpy.concatenate((d_y,d_y))
        d_x = numpy.concatenate((2*xmin-d_x,d_x))
    elif xmax is not None:
        d_y = numpy.concatenate((d_y,d_y))
        d_x = numpy.concatenate((d_x,2*xmax-d_x))
        
    if ymin is not None and ymax is not None:
        ymed = (ymin+ymax)/2
        d_x = numpy.concatenate((d_x[d_y<ymed],d_x,d_x[d_y>=ymed]))
        d_y = numpy.concatenate((2*ymin-d_y[d_y<ymed],d_y,2*ymax-d_y[d_y>=ymed]))
    elif ymin is not None:
        d_x = numpy.concatenate((d_x,d_x))
        d_y = numpy.concatenate((2*ymin-d_y,d_y))
    elif ymax is not None:
        d_x = numpy.concatenate((d_x,d_x))
        d_y = numpy.concatenate((d_y,2*ymax-d_y))

    return d_x, d_y

def kde_1d(d, xmin=None, xmax=None):
    axisExpansionFactor = xmax is None or xmin is None
    d_ = mirror_1d(d, xmin, xmax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, x = fastKDE.pdf(d_,axisExpansionFactor=axisExpansionFactor)
    p *= 2-axisExpansionFactor
    return x, p


def kde_2d(d_x, d_y, xmin=None, xmax=None, ymin=None, ymax=None):
    axisExpansionFactor=[xmax is None or xmin is None,
                         ymax is None or ymin is None] 
    d_x_, d_y_ = mirror_2d(d_x, d_y, xmin, xmax, ymin, ymax)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, (x, y) = fastKDE.pdf(d_x_, d_y_, axisExpansionFactor=axisExpansionFactor)

    p *= (2-axisExpansionFactor[0])
    p *= (2-axisExpansionFactor[1])

    return x, y, p
