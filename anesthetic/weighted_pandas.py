"""Pandas DataFrame and Series with weighted samples."""

import numpy as np
from pandas import Series, DataFrame, concat
from pandas.util import hash_pandas_object
from numpy.ma import masked_array
from anesthetic.utils import (compress_weights, channel_capacity, quantile,
                              temporary_seed)
from inspect import signature


class _WeightedObject(object):
    """Common methods for WeightedSeries and WeightedDataFrame."""

    def __init__(self, *args, **kwargs):
        weights = kwargs.pop('weights', None)
        super().__init__(*args, **kwargs)
        if weights is not None and 'weights' not in self.index.names:
            self._set_weights(weights)

    @property
    def weights(self):
        """Sample weights."""
        if 'weights' in self.index.names:
            return self.index.get_level_values('weights').to_numpy()
        else:
            return np.ones_like(self.index)

    @weights.setter
    def weights(self, weights):
        self._set_weights(weights)

    def _set_weights(self, weights):
        self.index = [self.index.get_level_values(name)
                      for name in self.index.names if name != 'weights'] \
                          + [weights]
        names = self.index.names[:-1] + ['weights']
        self.index.set_names(names, inplace=True)

    def reset_index(self, level=None, drop=False, inplace=False, *args):
        weights = self.weights
        answer = super().reset_index(level=level, drop=drop,
                                     inplace=inplace, *args)
        if inplace:
            self._set_weights(weights)
        else:
            answer._set_weights(weights)
            return answer

    def std(self, *args, **kwargs):
        """Weighted standard deviation of the sampled distribution."""
        return np.sqrt(self.var(*args, **kwargs))

    def kurtosis(self, *args, **kwargs):
        """Weighted kurtosis of the sampled distribution."""
        return self.kurt(*args, **kwargs)

    def median(self, *args, **kwargs):
        """Weighted median of the sampled distribution."""
        return self.quantile(*args, **kwargs)



class WeightedSeries(_WeightedObject, Series):
    """Weighted version of pandas.Series."""

    @property
    def _rand(self):
        """Random number for consistent compression."""
        seed = hash_pandas_object(self.index).sum() % 2**32
        with temporary_seed(seed):
            return np.random.rand(len(self))

    @property
    def weighted(self):
        return 'weights' in self.index.names

    def sample(self, *args, **kwargs):
        """Weighted sample."""
        return super().sample(weights=self.weights, *args, **kwargs)

    def mean(self, skipna=True):
        """Weighted mean of the sampled distribution."""
        null = self.isnull() & skipna
        return np.average(masked_array(self, null), weights=self.weights)

    def neff(self):
        """Effective number of samples."""
        return channel_capacity(self.weights)

    def var(self, skipna=True):
        """Weighted variance of the sampled distribution."""
        null = self.isnull() & skipna
        mean = self.mean(skipna=skipna)
        if np.isnan(mean):
            return mean
        return np.average(masked_array((self-mean)**2, null),
                          weights=self.weights)

    def cov(self, other, skipna=True):
        """Weighted covariance with another Series."""
        null = (self.isnull() | other.isnull()) & skipna
        x = self.mean(skipna=skipna)
        y = other.mean(skipna=skipna)
        if np.isnan(x) or np.isnan(y):
            return np.nan
        return np.average(masked_array((self-x)*(other-y), null),
                          weights=self.weights)

    def corr(self, other, skipna=True):
        """Weighted pearson correlation with another Series."""
        norm = self.std(skipna=skipna)*other.std(skipna=skipna)
        return self.cov(other, skipna=skipna)/norm

    def kurt(self, skipna=True):
        """Weighted kurtosis of the sampled distribution."""
        null = self.isnull() & skipna
        mean = self.mean(skipna=skipna)
        std = self.std(skipna=skipna)
        if np.isnan(mean) or np.isnan(std):
            return np.nan
        return np.average(masked_array(((self-mean)/std)**4, null),
                          weights=self.weights)

    def skew(self, skipna=True):
        """Weighted skewness of the sampled distribution."""
        null = self.isnull() & skipna
        mean = self.mean(skipna=skipna)
        std = self.std(skipna=skipna)
        if np.isnan(mean) or np.isnan(std):
            return np.nan
        return np.average(masked_array(((self-mean)/std)**3, null),
                          weights=self.weights)

    def mad(self, skipna=True):
        """Weighted mean absolute deviation of the sampled distribution."""
        null = self.isnull() & skipna
        mean = self.mean(skipna=skipna)
        if np.isnan(mean):
            return np.nan
        return np.average(masked_array(abs(self-mean), null),
                          weights=self.weights)

    def sem(self, skipna=True):
        """Weighted standard error of the mean."""
        return np.sqrt(self.var(skipna=skipna)/self.neff())

    def quantile(self, q=0.5, numeric_only=True, interpolation='linear'):
        """Weighted quantile of the sampled distribution."""
        if not numeric_only:
            raise NotImplementedError("numeric_only kwarg not implemented")
        return quantile(self.to_numpy(), q, self.weights, interpolation)

    def compress(self, nsamples=None):
        """Reduce the number of samples by discarding low-weights.

        Parameters
        ----------
        neff: int, optional
            effective number of samples after compression. If not supplied,
            then reduce to the channel capacity (theoretical optimum
            compression). If <=0, then compress so that all weights are unity.

        """
        i = compress_weights(self.weights, self._rand, nsamples)
        return self.repeat(i)

    @property
    def _constructor(self):
        return WeightedSeries

    @property
    def _constructor_expanddim(self):
        return WeightedDataFrame


class WeightedDataFrame(_WeightedObject, DataFrame):
    """Weighted version of pandas.DataFrame."""

    def _rand(self, axis):
        """Random number for consistent compression."""
        seed = hash_pandas_object(self._get_axis(axis)).sum() % 2**32
        with temporary_seed(seed):
            return np.random.rand(self.shape[axis])

    def _weights(self, axis):
        return self._get_axis(axis).get_level_values('weights').to_numpy()

    def weighted(self, axis):
        return 'weights' in self._get_axis(axis).names

    def neff(self, axis=0):
        """Effective number of samples."""
        if self.weighted(axis):
            return channel_capacity(self._weights(axis))
        else:
            return self.shape[axis]

    def sample(self, *args, **kwargs):
        """Weighted sample."""
        sig = signature(DataFrame.sample)
        axis = sig.bind(self, *args, **kwargs).arguments.get('axis', 0)
        if self.weighted(axis):
            return super().sample(weights=self._weights(axis), *args, **kwargs)
        else:
            return super().sample(*args, **kwargs)

    def mean(self, axis=0, skipna=True, *args, **kwargs):
        """Weighted mean of the sampled distribution."""
        if self.weighted(axis):
            null = self.isnull() & skipna
            mean = np.average(masked_array(self, null),
                              weights=self._weights(axis), axis=axis)
            return WeightedSeries(mean, index=self._get_axis(1-axis))
        else:
            return super().mean(axis=axis, skipna=skipna, *args, **kwargs)

    def var(self, axis=0, skipna=True, *args, **kwargs):
        """Weighted variance of the sampled distribution."""
        if self.weighted(axis):
            null = self.isnull() & skipna
            mean = self.mean(axis=axis, skipna=skipna)
            var = np.average(masked_array((self-mean)**2, null),
                             weights=self._weights(axis), axis=axis)
            return WeightedSeries(var, index=self._get_axis(1-axis))
        else:
            return super().var(axis=axis, skipna=skipna, *args, **kwargs)

    def cov(self, skipna=True, *args, **kwargs):
        """Weighted covariance of the sampled distribution."""
        if self.weighted(0):
            null = self.isnull() & skipna
            mean = self.mean(skipna=skipna)
            x = masked_array(self - mean, null)
            cov = np.ma.dot(self._weights(0)*x.T, x) / self._weights(0).sum().T
            return WeightedDataFrame(cov, index=self.columns,
                                     columns=self.columns)
        else:
            return super().cov(*args, **kwargs)

    def corr(self, skipna=True, *args, **kwargs):
        """Weighted pearson correlation matrix of the sampled distribution."""
        if self.weighted(0):
            cov = self.cov()
            diag = np.sqrt(np.diag(cov))
            return cov.divide(diag, axis=1).divide(diag, axis=0)
        else:
            return super().corr(*args, **kwargs)

    def corrwith(self, other, drop=False, *args, **kwargs):
        """Pairwise weighted pearson correlation."""
        if self.weighted(0):
            if isinstance(other, Series):
                answer = self.apply(lambda x: other.corr(x), axis=0)
                return WeightedSeries(answer)

            left, right = self.align(other, join="inner", copy=False)

            # mask missing values
            left = left + right * 0
            right = right + left * 0

            # demeaned data
            ldem = left - left.mean()
            rdem = right - right.mean()

            num = (ldem * rdem * self._weights(0)[:, None]).sum()
            dom = self._weights(0).sum() * left.std() * right.std()

            correl = num / dom

            if not drop:
                # Find non-matching labels along the given axis
                result_index = self._get_axis(1).union(other._get_axis(1))
                idx_diff = result_index.difference(correl.index)

                if len(idx_diff) > 0:
                    correl = concat([correl, Series([np.nan] * len(idx_diff),
                                                    index=idx_diff)])

            return WeightedSeries(correl)
        else:
            return super().corrwith(other, drop=drop, *args, **kwargs)

    def kurt(self, axis=0, skipna=True, *args, **kwargs):
        """Weighted kurtosis of the sampled distribution."""
        if self.weighted(axis):
            null = self.isnull() & skipna
            mean = self.mean(axis=axis, skipna=skipna)
            std = self.std(axis=axis, skipna=skipna)
            kurt = np.average(masked_array(((self-mean)/std)**4, null),
                              weights=self._weights(axis), axis=axis)
            return WeightedSeries(kurt, index=self._get_axis(1-axis))
        else:
            return super().kurt(axis=axis, skipna=skipna, *args, **kwargs)

    def skew(self, axis=0, skipna=True, *args, **kwargs):
        """Weighted skewness of the sampled distribution."""
        if self.weighted(axis):
            null = self.isnull() & skipna
            mean = self.mean(axis=axis, skipna=skipna)
            std = self.std(axis=axis, skipna=skipna)
            skew = np.average(masked_array(((self-mean)/std)**3, null),
                              weights=self._weights(axis), axis=axis)
            return WeightedSeries(skew, index=self._get_axis(1-axis))
        else:
            return super().skew(axis=axis, skipna=skipna, *args, **kwargs)

    def mad(self, axis=0, skipna=True, *args, **kwargs):
        """Weighted mean absolute deviation of the sampled distribution."""
        if self.weighted(axis):
            null = self.isnull() & skipna
            mean = self.mean(axis=axis, skipna=skipna)
            mad = np.average(masked_array(abs(self-mean), null),
                             weights=self._weights(axis), axis=axis)
            return WeightedSeries(mad, index=self._get_axis(1-axis))
        else:
            return super().var(axis=axis, skipna=skipna, *args, **kwargs)

    def sem(self, axis=0, skipna=True):
        """Weighted standard error of the mean."""
        n = self.neff(axis)
        return np.sqrt(self.var(axis=axis, skipna=skipna)/n)

    def quantile(self, q=0.5, axis=0, numeric_only=True,
                 interpolation='linear'):
        """Weighted quantile of the sampled distribution."""
        if not numeric_only:
            raise NotImplementedError("numeric_only kwarg not implemented")
        if self.weighted(axis):
            data = np.array([c.quantile(q, interpolation=interpolation,
                                        numeric_only=numeric_only)
                             for _, c in self.iteritems()])
            if np.isscalar(q):
                return WeightedSeries(data, index=self._get_axis(1-axis))
            else:
                return WeightedDataFrame(data.T, index=q,
                                         columns=self._get_axis(1-axis))
        else:
            return super().quantile(q=q, axis=axis, numeric_only=numeric_only,
                                    interpolation=interpolation)

    def compress(self, nsamples=None, axis=0):
        """Reduce the number of samples by discarding low-weights.

        Parameters
        ----------
        neff: int, optional
            effective number of samples after compression. If not supplied,
            then reduce to the channel capacity (theoretical optimum
            compression). If <=0, then compress so that all weights are unity.

        """
        if self.weighted(axis):
            i = compress_weights(self._weights(axis), self._rand(axis),
                                 nsamples)
            data = np.repeat(self.to_numpy(), i, axis=axis)
            i = self._get_axis(axis).repeat(i).droplevel('weights')
            df = WeightedDataFrame(data=data)
            df.set_axis(i, axis=axis, inplace=True)
            df.set_axis(self._get_axis(1-axis), axis=1-axis, inplace=True)
            return df
        else:
            return self

    @property
    def _constructor_sliced(self):
        return WeightedSeries

    @property
    def _constructor(self):
        return WeightedDataFrame
