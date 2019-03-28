import numpy

def channel_capacity(w):
    with numpy.errstate(divide='ignore'):
        return numpy.exp(numpy.nansum((numpy.log(sum(w))-numpy.log(w))*w)/sum(w))


def compress_weights(w, u=None, unity=False):
    """ Compresses weights to their approximate channel capacity."""
    if u is None:
        u = numpy.random.rand(len(w))

    if unity:
        W = w/w.max()
    else:
        W = w * channel_capacity(w) / w.sum()

    fraction, integer = numpy.modf(W)
    extra = (u<fraction).astype(int)
    return integer + extra
