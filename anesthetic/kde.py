import numpy
from fastkde import fastKDE

def mirror(d, xmin=None, xmax=None):
    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        return numpy.concatenate((2*xmin-d[d<xmed],d,2*xmax-d[d>=xmed]))
    elif xmin is not None:
        return numpy.concatenate((2*xmin-d,d))
    elif xmax is not None:
        return numpy.concatenate((d,2*xmax-d))
    else:
        return d

def kde_1d(d, xmin=None, xmax=None):
    axisExpansionFactor = xmax is None or xmin is None
    d_ = mirror(d, xmin, xmax)
    p, x = fastKDE.pdf(d_,axisExpansionFactor=axisExpansionFactor)
    p *= 2-axisExpansionFactor

    return x, p


def kde_2d(d_x, d_y, xmin=None, xmax=None, ymin=None, ymax=None):
    axisExpansionFactor=[1.,1.]
    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        d_x_ = numpy.concatenate((2*xmin-d_x[d_x<xmed],d,2*xmax-d_x[d_x>=xmed]))
        axisExpansionFactor=0.
    elif xmin is not None:
        d_x_ = numpy.concatenate((2*xmin-d_x,d_x))
    elif xmax is not None:
        d_x_ = numpy.concatenate((d_x,2*xmax-d_x))
    else:
        d_x_ = d_x

    p, x = fastKDE.pdf(d_,axisExpansionFactor=axisExpansionFactor)

    if xmin is not None and xmax is not None:
        p = 2*p[numpy.logical_and(x>=xmin,x<=xmax)]
        x = x[numpy.logical_and(x>=xmin,x<=xmax)]
    elif xmin is not None:
        p = 2*p[x>=xmin]
        x = x[x>=xmin]
    elif xmax is not None:
        p = 2*p[x<=xmax]
        x = x[x<=xmax]

    return x, p
