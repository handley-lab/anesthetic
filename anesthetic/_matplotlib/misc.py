import pandas.plotting._matplotlib.misc as misc
from anesthetic.weighted_pandas import _WeightedObject


def scatter_matrix(frame, *args, **kwargs):
    # noqa: disable=D103
    if isinstance(frame, _WeightedObject):
        frame = frame.compress(kwargs.pop('ncompress', None))
    return misc.scatter_matrix(frame, *args, **kwargs)


def bootstrap_plot(series, *args, **kwargs):
    # noqa: disable=D103
    if isinstance(series, _WeightedObject):
        series = series.compress(kwargs.pop('ncompress', None))
    return misc.bootstrap_plot(series, *args, **kwargs)
