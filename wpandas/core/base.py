import numpy as np
from pandas import MultiIndex
from pandas.util import hash_pandas_object
from wpandas.core.util.random import temporary_seed
from wpandas.core.util.weights import channel_capacity


class _WeightedObject(object):
    """Common methods for `WeightedSeries` and `WeightedDataFrame`.

    :meta public:
    """

    def __init__(self, *args, **kwargs):
        weights = kwargs.pop('weights', None)
        super().__init__(*args, **kwargs)
        if weights is not None:
            self.set_weights(weights, inplace=True)

    def isweighted(self, axis=0):
        """Determine if weights are actually present."""
        return 'weights' in self._get_axis(axis).names

    def get_weights(self, axis=0):
        """Retrieve sample weights from an axis."""
        if self.isweighted(axis):
            return self._get_axis(axis).get_level_values('weights').to_numpy()
        else:
            return np.ones_like(self._get_axis(axis))

    def drop_weights(self, axis=0):
        """Drop weights."""
        if self.isweighted(axis):
            return self.droplevel('weights', axis)
        return self.copy().__finalize__(self, "drop_weights")

    def set_weights(self, weights, axis=0, inplace=False, level=None):
        """Set sample weights along an axis.

        Parameters
        ----------
        weights : 1d array-like
            The sample weights to put in an index.

        axis : int (0,1), default=0
            Whether to put weights in an index or column.

        inplace : bool, default=False
            Whether to operate inplace, or return a new array.

        level : int
            Which level in the index to insert before.
            Defaults to inserting at back

        """
        if inplace:
            result = self
        else:
            result = self.copy()

        if weights is None:
            if result.isweighted(axis=axis):
                result = result.drop_weights(axis)
        else:
            names = [n for n in result._get_axis(axis).names if n != 'weights']
            index = [result._get_axis(axis).get_level_values(n) for n in names]
            if level is None:
                if result.isweighted(axis):
                    level = result._get_axis(axis).names.index('weights')
                else:
                    level = len(index)
            index.insert(level, weights)
            names.insert(level, 'weights')

            index = MultiIndex.from_arrays(index, names=names)
            result = result.set_axis(index, axis=axis, copy=False)

        if inplace:
            self._update_inplace(result)
        else:
            return result.__finalize__(self, "set_weights")

    def _rand(self, axis=0):
        """Random number for consistent compression."""
        seed = hash_pandas_object(self._get_axis(axis)).sum() % 2**32
        with temporary_seed(seed):
            return np.random.rand(self.shape[axis])

    def reset_index(self, level=None, drop=False, inplace=False,
                    *args, **kwargs):
        """Reset the index, retaining weights."""
        weights = self.get_weights()
        answer = super().reset_index(level=level, drop=drop,
                                     inplace=False, *args, **kwargs)
        answer.set_weights(weights, inplace=True)
        if inplace:
            self._update_inplace(answer)
        else:
            return answer.__finalize__(self, "reset_index")

    def neff(self, axis=0):
        """Effective number of samples."""
        if self.isweighted(axis):
            return channel_capacity(self.get_weights(axis))
        else:
            return self.shape[axis]
