"""General utilities."""
import numpy


def histogram(a, **kwargs):
    """ Produce a histogram for path-based plotting

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
