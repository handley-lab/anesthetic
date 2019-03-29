import numpy

def channel_capacity(w):
    with numpy.errstate(divide='ignore'):
        return numpy.exp(numpy.nansum((numpy.log(sum(w))-numpy.log(w))*w)/sum(w))


def compress_weights(w, u=None, nsamples=None):
    """ Compresses weights to their approximate channel capacity."""
    if u is None:
        u = numpy.random.rand(len(w))

    if nsamples is not None:
        W = w/w.max()
        if nsamples>0 and sum(W) > nsamples:
            W = W/sum(W)*nsamples
    else:
        W = w * channel_capacity(w) / w.sum()

    fraction, integer = numpy.modf(W)
    extra = (u<fraction).astype(int)
    return integer + extra
