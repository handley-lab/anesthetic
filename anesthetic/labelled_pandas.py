"""Pandas DataFrame and Series with labelled columns."""
from pandas import Series, DataFrame, MultiIndex
from pandas.core.indexing import (_LocIndexer as _LocIndexer_,
                                  _AtIndexer as _AtIndexer_)
import numpy as np
from functools import cmp_to_key


def ac(funcs, *args):
    """Accessor function helper.

    Given a list of callables `funcs`, and their arguments `*args`, evaluate
    each of these, catching exceptions, and then sort results by their
    dimensionality, smallest first. Return the non-exceptional result with the
    smallest dimensionality.
    """
    results = []
    errors = []
    for f in funcs:
        try:
            results.append(f(*args))
        except Exception as e:
            errors.append(e)

    def cmp(x, y):
        return 1 if x.ndim > y.ndim else -1

    results.sort(key=cmp_to_key(cmp))

    for s in results:
        if s is not None:
            return s
    raise errors[-1]


class _LocIndexer(_LocIndexer_):
    def __getitem__(self, key):
        return ac([_AtIndexer_("loc", self.obj.drop_labels(i)).__getitem__
                   for i in self.obj._all_axes()] + [super().__getitem__], key)


class _AtIndexer(_AtIndexer_):
    def __getitem__(self, key):
        return ac([_AtIndexer_("at", self.obj.drop_labels(i)).__getitem__
                   for i in self.obj._all_axes()] + [super().__getitem__], key)


class _LabelledObject(object):
    """Common methods for LabelledSeries and LabelledDataFrame."""

    def __init__(self, *args, **kwargs):
        self._labels = ("labels", "labels")
        labels = kwargs.pop(self._labels[0], None)
        super().__init__(*args, **kwargs)
        if labels is not None:
            self.set_labels(labels, inplace=True)

    def islabelled(self, axis=0):
        """Determine if labels are actually present."""
        return self._labels[axis] in self._get_axis(axis).names

    def get_labels(self, axis=0):
        """Retrieve labels from an axis."""
        if self.islabelled(axis):
            return self._get_axis(axis).get_level_values(
                    self._labels[axis]).to_numpy()
        else:
            return None

    def get_labels_map(self, axis=0):
        """Retrieve mapping from paramnames to labels from an axis."""
        labels = self.get_labels(axis)
        params = self.drop_labels(axis)._get_axis(axis)
        if labels is None:
            labels = params
        return dict(zip(params, labels))

    def get_label(self, param, axis=0):
        """Retrieve mapping from paramnames to labels from an axis."""
        return self.get_labels_map(axis)[param]

    def drop_labels(self, axis=0):
        axes = np.atleast_1d(axis)
        result = self
        for axis in axes:
            if self.islabelled(axis):
                result = result.droplevel(self._labels[axis], axis)
        return result

    def _all_axes(self):
        if isinstance(self, LabelledSeries):
            return [0]
        else:
            return [0, 1, [0, 1]]

    @property
    def loc(self):
        return _LocIndexer("loc", self)

    @property
    def at(self):
        return _AtIndexer("at", self)

    def xs(self, key, axis=0, level=None, drop_level=True):
        return ac([super(_LabelledObject, self.drop_labels(i)).xs
                   for i in self._all_axes()] + [super().xs],
                  key, axis, level, drop_level)

    def __getitem__(self, key):
        return ac([super(_LabelledObject, self.drop_labels(i)).__getitem__
                   for i in self._all_axes()] + [super().__getitem__], key)

    def __setitem__(self, key, val):
        super().__setitem__(key, val)

    def set_labels(self, labels, axis=0, inplace=False, level=None):
        """Set labels along an axis."""
        if inplace:
            result = self
        else:
            result = self.copy()

        if labels is None:
            if result.islabelled(axis=axis):
                result = result.drop_labels(axis)
        else:
            names = [n for n in result._get_axis(axis).names
                     if n != self._labels[axis]]
            index = [result._get_axis(axis).get_level_values(n) for n in names]
            if level is None:
                if result.islabelled(axis):
                    level = result._get_axis(axis
                                             ).names.index(self._labels[axis])
                else:
                    level = len(index)
            index.insert(level, labels)
            names.insert(level, self._labels[axis])

            index = MultiIndex.from_arrays(index, names=names)
            result.set_axis(index, axis=axis, inplace=True)

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

    _metadata = Series._metadata + ['_labels']

    @property
    def _constructor(self):
        return LabelledSeries

    @property
    def _constructor_expanddim(self):
        return LabelledDataFrame


class LabelledDataFrame(_LabelledObject, DataFrame):
    """Labelled version of pandas.DataFrame."""

    _metadata = DataFrame._metadata + ['_labels']

    @property
    def _constructor_sliced(self):
        return LabelledSeries

    @property
    def _constructor(self):
        return LabelledDataFrame

    def transpose(self, copy=False):
        """Transpose."""
        result = super().transpose(copy=copy)
        result._labels = (result._labels[1], result._labels[0])
        return result
