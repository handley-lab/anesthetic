"""Pandas DataFrame and Series with weighted samples."""

from inspect import signature
import numpy as np
from pandas import Series, DataFrame, concat, MultiIndex
from pandas.util import hash_pandas_object
from numpy.ma import masked_array
from anesthetic.utils import (compress_weights, channel_capacity, quantile,
                              temporary_seed, adjust_docstrings)


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


class WeightedSeries(_WeightedObject, Series):
    """Weighted version of :class:`pandas.Series`."""

    def mean(self, skipna=True):  # noqa: D102
        null = self.isnull() & skipna
        return np.average(masked_array(self, null), weights=self.get_weights())

    def std(self, *args, **kwargs):  # noqa: D102
        return np.sqrt(self.var(*args, **kwargs))

    def kurtosis(self, *args, **kwargs):  # noqa: D102
        return self.kurt(*args, **kwargs)

    def median(self, *args, **kwargs):  # noqa: D102
        return self.quantile(*args, **kwargs)

    def var(self, skipna=True):  # noqa: D102
        null = self.isnull() & skipna
        mean = self.mean(skipna=skipna)
        if np.isnan(mean):
            return mean
        return np.average(masked_array((self-mean)**2, null),
                          weights=self.get_weights())

    def cov(self, other, skipna=True):  # noqa: D102
        null = (self.isnull() | other.isnull()) & skipna
        x = self.mean(skipna=skipna)
        y = other.mean(skipna=skipna)
        if np.isnan(x) or np.isnan(y):
            return np.nan
        return np.average(masked_array((self-x)*(other-y), null),
                          weights=self.get_weights())

    def corr(self, other, method="pearson", skipna=True):  # noqa: D102
        norm = self.std(skipna=skipna)*other.std(skipna=skipna)
        return self.cov(other, skipna=skipna)/norm

    def kurt(self, skipna=True):  # noqa: D102
        null = self.isnull() & skipna
        mean = self.mean(skipna=skipna)
        std = self.std(skipna=skipna)
        if np.isnan(mean) or np.isnan(std):
            return np.nan
        return np.average(masked_array(((self-mean)/std)**4, null),
                          weights=self.get_weights())

    def skew(self, skipna=True):  # noqa: D102
        null = self.isnull() & skipna
        mean = self.mean(skipna=skipna)
        std = self.std(skipna=skipna)
        if np.isnan(mean) or np.isnan(std):
            return np.nan
        return np.average(masked_array(((self-mean)/std)**3, null),
                          weights=self.get_weights())

    def mad(self, skipna=True):  # noqa: D102
        null = self.isnull() & skipna
        mean = self.mean(skipna=skipna)
        if np.isnan(mean):
            return np.nan
        return np.average(masked_array(abs(self-mean), null),
                          weights=self.get_weights())

    def sem(self, skipna=True):  # noqa: D102
        return np.sqrt(self.var(skipna=skipna)/self.neff())

    def quantile(self, q=0.5, interpolation='linear'):  # noqa: D102
        return quantile(self.to_numpy(), q, self.get_weights(), interpolation)

    def compress(self, ncompress=True):
        """Reduce the number of samples by discarding low-weights.

        Parameters
        ----------
        ncompress : int, optional
            effective number of samples after compression. If not supplied
            (or True), then reduce to the channel capacity (theoretical optimum
            compression). If <=0, then compress so that all weights are unity.

        """
        i = compress_weights(self.get_weights(), self._rand(), ncompress)
        return self.repeat(i)

    def sample(self, *args, **kwargs):  # noqa: D102
        return super().sample(weights=self.get_weights(), *args, **kwargs)

    @property
    def _constructor(self):
        return WeightedSeries

    @property
    def _constructor_expanddim(self):
        return WeightedDataFrame


class WeightedDataFrame(_WeightedObject, DataFrame):
    """Weighted version of :class:`pandas.DataFrame`."""

    def mean(self, axis=0, skipna=True, *args, **kwargs):  # noqa: D102
        if self.isweighted(axis):
            null = self.isnull() & skipna
            mean = np.average(masked_array(self, null),
                              weights=self.get_weights(axis), axis=axis)
            return self._constructor_sliced(mean, index=self._get_axis(1-axis))
        else:
            return super().mean(axis=axis, skipna=skipna, *args, **kwargs)

    def std(self, *args, **kwargs):  # noqa: D102
        return np.sqrt(self.var(*args, **kwargs))

    def kurtosis(self, *args, **kwargs):  # noqa: D102
        return self.kurt(*args, **kwargs)

    def median(self, *args, **kwargs):  # noqa: D102
        return self.quantile(*args, **kwargs)

    def var(self, axis=0, skipna=True, *args, **kwargs):  # noqa: D102
        if self.isweighted(axis):
            null = self.isnull() & skipna
            mean = self.mean(axis=axis, skipna=skipna)
            var = np.average(masked_array((self-mean)**2, null),
                             weights=self.get_weights(axis), axis=axis)
            return self._constructor_sliced(var, index=self._get_axis(1-axis))
        else:
            return super().var(axis=axis, skipna=skipna, *args, **kwargs)

    def cov(self, skipna=True, *args, **kwargs):  # noqa: D102
        if self.isweighted():
            null = self.isnull() & skipna
            mean = self.mean(skipna=skipna)
            x = masked_array(self - mean, null)
            cov = np.ma.dot(self.get_weights()*x.T, x) \
                / self.get_weights().sum().T
            if kwargs:
                raise NotImplementedError("The keywords %s are not implemented"
                                          "for the calculation of the"
                                          "covariance with weighted samples."
                                          % kwargs)
            return self._constructor(cov, index=self.columns,
                                     columns=self.columns)
        else:
            return super().cov(*args, **kwargs)

    def corr(self, method="pearson", skipna=True,
             *args, **kwargs):  # noqa: D102
        if self.isweighted():
            cov = self.cov()
            diag = np.sqrt(np.diag(cov))
            return cov.divide(diag, axis=1).divide(diag, axis=0)
        else:
            return super().corr(*args, **kwargs)

    def corrwith(self, other, axis=0, drop=False, method="pearson",
                 *args, **kwargs):  # noqa: D102
        if self.isweighted(axis):
            if isinstance(other, Series):
                answer = self.apply(lambda x: other.corr(x, method=method),
                                    axis=axis)
                return self._constructor_sliced(answer)

            left, right = self.align(other, join="inner", copy=False)
            weights = self.get_weights(axis)

            if axis == 1:
                left = left.T
                right = right.T

            # mask missing values
            left = left + right * 0
            right = right + left * 0

            # demeaned data
            ldem = left - left.mean()
            rdem = right - right.mean()

            num = (ldem * rdem * weights[:, None]).sum()
            dom = weights.sum() * left.std() * right.std()

            correl = num / dom

            if not drop:
                # Find non-matching labels along the given axis
                result_index = self._get_axis(1-axis).union(
                    other._get_axis(1-axis))
                idx_diff = result_index.difference(correl.index)

                if len(idx_diff) > 0:
                    correl = concat([correl, Series([np.nan] * len(idx_diff),
                                                    index=idx_diff)])

            return self._constructor_sliced(correl)
        else:
            return super().corrwith(other, drop=drop, axis=axis, method=method,
                                    *args, **kwargs)

    def kurt(self, axis=0, skipna=True, *args, **kwargs):  # noqa: D102
        if self.isweighted(axis):
            null = self.isnull() & skipna
            mean = self.mean(axis=axis, skipna=skipna)
            std = self.std(axis=axis, skipna=skipna)
            kurt = np.average(masked_array(((self-mean)/std)**4, null),
                              weights=self.get_weights(axis), axis=axis)
            return self._constructor_sliced(kurt, index=self._get_axis(1-axis))
        else:
            return super().kurt(axis=axis, skipna=skipna, *args, **kwargs)

    def skew(self, axis=0, skipna=True, *args, **kwargs):  # noqa: D102
        if self.isweighted(axis):
            null = self.isnull() & skipna
            mean = self.mean(axis=axis, skipna=skipna)
            std = self.std(axis=axis, skipna=skipna)
            skew = np.average(masked_array(((self-mean)/std)**3, null),
                              weights=self.get_weights(axis), axis=axis)
            return self._constructor_sliced(skew, index=self._get_axis(1-axis))
        else:
            return super().skew(axis=axis, skipna=skipna, *args, **kwargs)

    def mad(self, axis=0, skipna=True, *args, **kwargs):  # noqa: D102
        if self.isweighted(axis):
            null = self.isnull() & skipna
            mean = self.mean(axis=axis, skipna=skipna)
            mad = np.average(masked_array(abs(self-mean), null),
                             weights=self.get_weights(axis), axis=axis)
            return self._constructor_sliced(mad, index=self._get_axis(1-axis))
        else:
            return super().var(axis=axis, skipna=skipna, *args, **kwargs)

    def sem(self, axis=0, skipna=True):  # noqa: D102
        n = self.neff(axis)
        return np.sqrt(self.var(axis=axis, skipna=skipna)/n)

    def quantile(self, q=0.5, axis=0, numeric_only=None,
                 interpolation='linear', method=None):  # noqa: D102
        if self.isweighted(axis):
            if numeric_only is not None or method is not None:
                raise NotImplementedError(
                    "`numeric_only` and `method` kwargs not implemented for "
                    "`WeightedSeries` and `WeightedDataFrame`."
                )
            data = np.array([c.quantile(q=q, interpolation=interpolation)
                             for _, c in self.items()])
            if np.isscalar(q):
                return self._constructor_sliced(data,
                                                index=self._get_axis(1-axis))
            else:
                return self._constructor(data.T, index=q,
                                         columns=self._get_axis(1-axis))
        else:
            if numeric_only is None:
                numeric_only = True
            if method is None:
                method = 'single'
            return super().quantile(q=q, axis=axis, numeric_only=numeric_only,
                                    interpolation=interpolation, method=method)

    def compress(self, ncompress=True, axis=0):
        """Reduce the number of samples by discarding low-weights.

        Parameters
        ----------
        ncompress : int, optional
            effective number of samples after compression. If not supplied
            (or True), then reduce to the channel capacity (theoretical optimum
            compression). If <=0, then compress so that all weights are unity.

        """
        if self.isweighted(axis):
            i = compress_weights(self.get_weights(axis), self._rand(axis),
                                 ncompress)
            data = np.repeat(self.to_numpy(), i, axis=axis)
            i = self.drop_weights(axis)._get_axis(axis).repeat(i)
            df = self._constructor(data=data)
            df = df.set_axis(i, axis=axis, copy=False)
            df = df.set_axis(self._get_axis(1-axis), axis=1-axis, copy=False)
            return df
        else:
            return self

    def sample(self, *args, **kwargs):  # noqa: D102
        sig = signature(DataFrame.sample)
        axis = sig.bind(self, *args, **kwargs).arguments.get('axis', 0)
        if self.isweighted(axis):
            return super().sample(weights=self.get_weights(axis),
                                  *args, **kwargs)
        else:
            return super().sample(*args, **kwargs)

    @property
    def _constructor_sliced(self):
        return WeightedSeries

    @property
    def _constructor(self):
        return WeightedDataFrame


for cls in [WeightedDataFrame, WeightedSeries]:
    adjust_docstrings(cls, r'\bDataFrame\b', 'WeightedDataFrame')
    adjust_docstrings(cls, r'\bDataFrames\b', 'WeightedDataFrames')
    adjust_docstrings(cls, r'\bSeries\b', 'WeightedSeries')
    adjust_docstrings(cls, 'core', 'pandas.core')
    adjust_docstrings(cls, 'DataFrameGroupBy',
                           'pandas.core.groupby.DataFrameGroupBy')
    adjust_docstrings(cls, 'SeriesGroupBy',
                           'pandas.core.groupby.SeriesGroupBy')
    adjust_docstrings(cls, 'pandas.core.window.Rolling.quantile',
                           'pandas.core.window.rolling.Rolling.quantile')
