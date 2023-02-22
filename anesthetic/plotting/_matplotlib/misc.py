import pandas.plotting._matplotlib.misc as misc
from anesthetic.plotting._matplotlib.core import _compress_weights


def scatter_matrix(frame, *args, **kwargs):
    # noqa: disable=D103
    frame = _compress_weights(kwargs, frame)
    return misc.scatter_matrix(frame, *args, **kwargs)


def bootstrap_plot(series, *args, **kwargs):
    # noqa: disable=D103
    series = _compress_weights(kwargs, series)
    return misc.bootstrap_plot(series, *args, **kwargs)
