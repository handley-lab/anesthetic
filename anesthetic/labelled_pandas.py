"""Pandas DataFrame and Series with weighted samples."""
from pandas import Series, DataFrame
from pandas.core.indexing import (_LocIndexer as _LocIndexer_,
                                  _AtIndexer as _AtIndexer_)


class _LocIndexer(_LocIndexer_):
    def __getitem__(self, key):
        return _LocIndexer_("loc", self.obj.drop_labels()).__getitem__(key)


class _AtIndexer(_AtIndexer_):
    def __getitem__(self, key):
        return _AtIndexer_("at", self.obj.drop_labels()).__getitem__(key)


class _LabelledObject(object):
    """Common methods for LabelledSeries and LabelledDataFrame."""

    _labels = "labels"

    def __init__(self, *args, **kwargs):
        labels = kwargs.pop(self._labels, None)
        super().__init__(*args, **kwargs)
        if labels is not None:
            self.set_labels(labels, inplace=True)

    def islabelled(self, axis=0):
        """Determine if labels are actually present."""
        return self._labels in self._get_axis(axis).names

    def get_labels(self, axis=0):
        """Retrieve labels from an axis."""
        if self.islabelled(axis):
            return self._get_axis(axis).get_level_values(
                    self._labels).to_numpy()
        else:
            return None

    def drop_labels(self):
        result = self
        for axis in range(self.ndim):
            if self.islabelled(axis):
                result = result.droplevel(self._labels, axis)
        return result

    @property
    def loc(self):
        return _LocIndexer("loc", self)

    @property
    def at(self):
        return _AtIndexer("at", self)

    def xs(self, key, axis=0, level=None, drop_level=True):
        return super(_LabelledObject, self.drop_labels()).xs(
                key, axis, level, drop_level)

    def __getitem__(self, key):
        return super(_LabelledObject, self.drop_labels()).__getitem__(key)

    def set_labels(self, labels, axis=0, inplace=False):
        """Set labels along an axis."""
        if labels is None:
            if self.islabelled(axis):
                result = self.drop_labels(axis)
            elif inplace:
                result = self
            else:
                result = self.copy()
        else:
            result = self.set_axis([self._get_axis(axis).get_level_values(name)
                                    for name in self._get_axis(axis).names
                                    if name != self._labels] + [labels],
                                   axis=axis)
            names = result._get_axis(axis).names[:-1] + [self._labels]
            result._get_axis(axis).set_names(names, inplace=True)

        if inplace:
            self._update_inplace(result)
        else:
            return result.__finalize__(self, "set_labels")

    def reset_index(self, level=None, drop=False, inplace=False,
                    *args, **kwargs):
        """Reset the index, retaining labels."""
        labels = self.get_labels()
        answer = super().reset_index(level=level, drop=drop,
                                     inplace=False, *args, **kwargs)
        answer.set_labels(labels, inplace=True)
        if inplace:
            self._update_inplace(answer)
        else:
            return answer.__finalize__(self, "reset_index")


class LabelledSeries(_LabelledObject, Series):
    """Labelled version of pandas.Series."""

    @property
    def _constructor(self):
        return LabelledSeries

    @property
    def _constructor_expanddim(self):
        return LabelledDataFrame


class LabelledDataFrame(_LabelledObject, DataFrame):
    """Labelled version of pandas.DataFrame."""

    @property
    def _constructor_sliced(self):
        return LabelledSeries

    @property
    def _constructor(self):
        return LabelledDataFrame
