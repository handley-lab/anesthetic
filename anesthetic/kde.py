import numpy
from fastkde import fastKDE

def kde_1d(d, xmin=None, xmax=None):
    axisExpansionFactor=1.
    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        d_ = numpy.concatenate((2*xmin-d[d<xmed],d,2*xmax-d[d>=xmed]))
        axisExpansionFactor=0.
    elif xmin is not None:
        d_ = numpy.concatenate((2*xmin-d,d))
    elif xmax is not None:
        d_ = numpy.concatenate((d,2*xmax-d))
    else:
        d_ = d

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
