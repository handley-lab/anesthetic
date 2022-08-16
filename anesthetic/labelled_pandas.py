"""Pandas DataFrame and Series with weighted samples."""

from pandas import Series, DataFrame

import pandas.core.indexing 

class _LocIndexer(pandas.core.indexing._LocIndexer):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = (self.obj.label(key[0], axis=0),
                   self.obj.label(key[1], axis=1))
        return super().__getitem__(key)



class _LabelledObject(object):
    """Common methods for LabelledSeries and LabelledDataFrame."""

    def label(self, x, axis=1):
        if self.islabelled(axis):
            frame = self._get_axis(axis).to_frame()
            frame.reset_index('labels', drop=True, inplace=True)
            frame = frame.apply(tuple, axis=axis) 
            frame['x0']
            return frame[x]
        else:
            return x

    def __init__(self, *args, **kwargs):
        labels = kwargs.pop('labels', None)
        super().__init__(*args, **kwargs)
        if labels is not None:
            self.set_labels(labels, inplace=True)

    def islabelled(self, axis=1):
        """Determine if labels are actually present."""
        return 'labels' in self._get_axis(axis).names

    def get_labels(self, axis=1):
        """Retrieve labels from an axis."""
        if self.islabelled(axis):
            return self._get_axis(axis).get_level_values('labels').to_numpy()
        else:
            return None

    @property
    def loc(self):
        return _LocIndexer("loc", self)

#    def __getitem__(self, x):
#        if x in self._get_axis(1).droplevel('labels'):
#            x = self.label(x)
#        return super().__getitem__(x)

    def set_labels(self, labels, axis=1, inplace=False):
        """Set labels along an axis."""
        if labels is None:
            if self.islabelled(axis=axis):
                result = self.droplevel('labels', axis=axis)
            elif inplace:
                result = self
            else:
                result = self.copy()
        else:
            result = self.set_axis([self._get_axis(axis).get_level_values(name)
                                    for name in self._get_axis(axis).names
                                    if name != 'labels'] + [labels],
                                   axis=axis)
            names = result._get_axis(axis).names[:-1] + ['labels']
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
