"""Pandas DataFrame and Series with weighted samples."""

import numpy as np
from pandas import Series, DataFrame
from pandas.util import hash_pandas_object
from numpy.ma import masked_array
from anesthetic.utils import (compress_weights, channel_capacity, quantile,
                              temporary_seed)


class _WeightedObject(object):
    """Common methods for WeightedSeries and WeightedDataFrame."""

    def __init__(self, *args, **kwargs):
        weights = kwargs.pop('weights', None)
        super().__init__(*args, **kwargs)
        self.weights = weights

    @property
    def weights(self):
        """Sample weights."""
        if self.index.nlevels == 1:
            return np.ones_like(self.index)
        else:
            return self.index.get_level_values('weights').to_numpy()

    @weights.setter
    def weights(self, weights):
        if weights is not None:
            self.index = [self.index.get_level_values(0), weights]
            self.index.set_names(['#', 'weights'], inplace=True)

    @property
    def _rand(self):
        """Random number for consistent compression."""
        seed = hash_pandas_object(self.index).sum() % 2**32
        with temporary_seed(seed):
            return np.random.rand(len(self))

    def std(self, *args, **kwargs):
        """Weighted standard deviation of the sampled distribution."""
        return np.sqrt(self.var(*args, **kwargs))

    def kurtosis(self, *args, **kwargs):
        """Weighted kurtosis of the sampled distribution."""
        return self.kurt(*args, **kwargs)

    def median(self, *args, **kwargs):
        """Weighted median of the sampled distribution."""
        return self.quantile(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """Weighted sample."""
        return super().sample(weights=self.weights, *args, **kwargs)

    def hist(self, *args, **kwargs):
        """Weighted histogram of the sampled distribution."""
        return super().hist(weights=self.weights, *args, **kwargs)

    def neff(self):
        """Effective number of samples."""
        return channel_capacity(self.weights)


class WeightedSeries(_WeightedObject, Series):
    """Weighted version of pandas.Series."""

    def mean(self, skipna=True):
        """Weighted mean of the sampled distribution."""
        null = self.isnull() & skipna
        return np.average(masked_array(self, null), weights=self.weights)

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

    def mean(self, axis=0, skipna=True):
        """Weighted mean of the sampled distribution."""
        if axis == 0:
            null = self.isnull() & skipna
            mean = np.average(masked_array(self, null),
                              weights=self.weights, axis=0)
            return Series(mean, index=self.columns)
        else:
            return super().mean(axis=axis, skipna=skipna)

    def var(self, axis=0, skipna=True):
        """Weighted variance of the sampled distribution."""
        if axis == 0:
            null = self.isnull() & skipna
            mean = self.mean(skipna=skipna).to_numpy()
            var = np.average(masked_array((self-mean)**2, null),
                             weights=self.weights, axis=0)
            return Series(var, index=self.columns)
        else:
            return super().var(axis=axis, skipna=skipna)

    def cov(self, skipna=True):
        """Weighted covariance of the sampled distribution."""
        null = self.isnull() & skipna
        mean = self.mean(skipna=skipna).to_numpy()
        x = masked_array(self - mean, null)
        cov = np.ma.dot(self.weights * x.T, x) / self.weights.sum().T
        return DataFrame(cov, index=self.columns, columns=self.columns)

    def corr(self, skipna=True):
        """Weighted pearson correlation matrix of the sampled distribution."""
        cov = self.cov()
        diag = np.sqrt(np.diag(cov))
        return cov.divide(diag, axis=1).divide(diag, axis=0)

    def corrwith(self, other, drop=False):
        """Pairwise weighted pearson correlation."""
        this = self._get_numeric_data()

        if isinstance(other, Series):
            return this.apply(lambda x: other.corr(x), axis=0)

        other = other._get_numeric_data()
        left, right = this.align(other, join="inner", copy=False)

        # mask missing values
        left = left + right * 0
        right = right + left * 0

        # demeaned data
        ldem = left - left.mean()
        rdem = right - right.mean()

        num = (ldem * rdem * self.weights[:, None]).sum()
        dom = self.weights.sum() * left.std() * right.std()

        correl = num / dom

        if not drop:
            # Find non-matching labels along the given axis
            result_index = this._get_axis(1).union(other._get_axis(1))
            idx_diff = result_index.difference(correl.index)

            if len(idx_diff) > 0:
                correl = correl.append(Series([np.nan] * len(idx_diff),
                                              index=idx_diff))

        return correl

    def kurt(self, axis=0, skipna=True):
        """Weighted kurtosis of the sampled distribution."""
        if axis == 0:
            null = self.isnull() & skipna
            mean = self.mean(skipna=skipna).to_numpy()
            std = self.std(skipna=skipna).to_numpy()
            kurt = np.average(masked_array(((self-mean)/std)**4, null),
                              weights=self.weights, axis=0)
            return Series(kurt, index=self.columns)
        else:
            return super().kurt(axis=axis, skipna=skipna)

    def skew(self, axis=0, skipna=True):
        """Weighted skewness of the sampled distribution."""
        if axis == 0:
            null = self.isnull() & skipna
            mean = self.mean(skipna=skipna).to_numpy()
            std = self.std(skipna=skipna).to_numpy()
            skew = np.average(masked_array(((self-mean)/std)**3, null),
                              weights=self.weights, axis=0)
            return Series(skew, index=self.columns)
        else:
            return super().skew(axis=axis, skipna=skipna)

    def mad(self, axis=0, skipna=True):
        """Weighted mean absolute deviation of the sampled distribution."""
        if axis == 0:
            null = self.isnull() & skipna
            mean = self.mean(skipna=skipna).to_numpy()
            mad = np.average(masked_array(abs(self-mean), null),
                             weights=self.weights, axis=0)
            return Series(mad, index=self.columns)
        else:
            return super().var(axis=axis, skipna=skipna)

    def sem(self, axis=0, skipna=True):
        """Weighted standard error of the mean."""
        n = self.neff() if axis == 0 else self.shape[1]
        return np.sqrt(self.var(axis=axis, skipna=skipna)/n)

    def quantile(self, q=0.5, axis=0, numeric_only=True,
                 interpolation='linear'):
        """Weighted quantile of the sampled distribution."""
        if not numeric_only:
            raise NotImplementedError("numeric_only kwarg not implemented")
        if axis == 0:
            data = np.array([c.quantile(q) for _, c in self.iteritems()])
            if np.isscalar(q):
                return Series(data, index=self.columns)
            else:
                return DataFrame(data.T, columns=self.columns, index=q)
        else:
            return super().quantile(q=q, axis=axis, numeric_only=numeric_only,
                                    interpolation=interpolation)

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
        data = np.repeat(self.to_numpy(), i, axis=0)
        index = self.index.repeat(i)
        df = DataFrame(data=data, index=index, columns=self.columns)
        df.index = df.index.get_level_values('#')
        return df

    @property
    def _constructor_sliced(self):
        return WeightedSeries

    @property
    def _constructor(self):
        return WeightedDataFrame
