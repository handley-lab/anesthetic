"""Pandas DataFrame with weights and labels."""
from anesthetic.weighted_pandas import WeightedDataFrame, WeightedSeries
from anesthetic.labelled_pandas import LabelledDataFrame, LabelledSeries
import pandas as pd


def read_csv(filename, *args, **kwargs):
    """Read a CSV file into a ``WeightedLabelledDataFrame``."""
    df = pd.read_csv(filename, index_col=[0, 1], header=[0, 1],
                     *args, **kwargs)
    wldf = WeightedLabelledDataFrame(df)
    if wldf.isweighted() and wldf.islabelled():
        wldf.set_weights(wldf.get_weights().astype(float), inplace=True)
        return wldf
    df = pd.read_csv(filename, index_col=[0, 1], *args, **kwargs)
    wldf = WeightedLabelledDataFrame(df)
    if wldf.isweighted():
        return wldf
    df = pd.read_csv(filename, index_col=0, header=[0, 1], *args, **kwargs)
    wldf = WeightedLabelledDataFrame(df)
    if wldf.islabelled():
        return wldf
    df = pd.read_csv(filename, index_col=0, *args, **kwargs)
    return WeightedLabelledDataFrame(df)


class WeightedLabelledDataFrame(WeightedDataFrame, LabelledDataFrame):
    """:class:`pandas.DataFrame` with weights and labels."""

    _metadata = WeightedDataFrame._metadata + LabelledDataFrame._metadata

    def __init__(self, *args, **kwargs):
        labels = kwargs.pop('labels', None)
        if not hasattr(self, '_labels'):
            self._labels = ('weights', 'labels')
        super().__init__(*args, **kwargs)
        if labels is not None:
            if isinstance(labels, dict):
                labels = [labels.get(p, '') for p in self]
            self.set_labels(labels, inplace=True)

    def islabelled(self, axis=1):
        """Search for existence of labels."""
        return super().islabelled(axis=axis)

    def get_labels(self, axis=1):
        """Retrieve labels from an axis."""
        return super().get_labels(axis=axis)

    def get_labels_map(self, axis=1, fill=True):
        """Retrieve mapping from paramnames to labels from an axis."""
        return super().get_labels_map(axis=axis, fill=fill)

    def get_label(self, param, axis=1):
        """Retrieve mapping from paramnames to labels from an axis."""
        return super().get_label(param, axis=axis)

    def set_label(self, param, value, axis=1):
        """Set a specific label to a specific value on an axis."""
        return super().set_label(param, value, axis=axis, inplace=True)

    def drop_labels(self, axis=1):
        """Drop the labels from an axis if present."""
        return super().drop_labels(axis)

    def set_labels(self, labels, axis=1, inplace=False, level=None):
        """Set labels along an axis."""
        return super().set_labels(labels, axis=axis,
                                  inplace=inplace, level=level)

    @property
    def _constructor(self):
        return WeightedLabelledDataFrame

    @property
    def _constructor_sliced(self):
        return WeightedLabelledSeries


class WeightedLabelledSeries(WeightedSeries, LabelledSeries):
    """Series with weights and labels."""

    _metadata = WeightedSeries._metadata + LabelledSeries._metadata

    def __init__(self, *args, **kwargs):
        if not hasattr(self, '_labels'):
            self._labels = ('weights', 'labels')
        super().__init__(*args, **kwargs)

    def set_label(self, param, value, axis=0):
        """Set a specific label to a specific value."""
        return super().set_label(param, value, axis=axis, inplace=True)

    @property
    def _constructor(self):
        return WeightedLabelledSeries

    @property
    def _constructor_expanddim(self):
        return WeightedLabelledDataFrame
