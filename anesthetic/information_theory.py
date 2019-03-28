import numpy

def channel_capacity(w):
    return numpy.exp(sum((numpy.log(sum(w))-numpy.log(w))*w)/sum(w))


def compress_weights(w):
    """ Compresses weights to their approximate channel capacity."""
    W = w * channel_capacity(w) / w.sum()
    fraction, integer = numpy.modf(W)
    extra = (numpy.random.rand(len(fraction))<fraction).astype(int)
    return integer + extra
