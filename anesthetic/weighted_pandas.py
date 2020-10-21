"""Pandas DataFrame and Series with weighted samples."""

import numpy as np
import pandas
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
        seed = pandas.util.hash_pandas_object(self.index).sum() % 2**32
        with temporary_seed(seed):
            return np.random.rand(len(self))

    def std(self, *args, **kwargs):
        """Weighted standard deviation of the sampled distribution."""
        return np.sqrt(self.var(*args, **kwargs))

    def median(self):
        """Weighted median of the sampled distribution."""
        return self.quantile()

    def hist(self, *args, **kwargs):
        """Weighted histogram of the sampled distribution."""
        return super().hist(weights=self.weights, *args, **kwargs)

    def neff(self):
        """Effective number of samples."""
        return channel_capacity(self.weights)


class WeightedSeries(_WeightedObject, pandas.Series):
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
        return np.average((masked_array(self, null)-mean)**2,
                          weights=self.weights)

    def quantile(self, q=0.5):
        """Weighted quantile of the sampled distribution."""
        return quantile(self.values, q, self.weights)

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


class WeightedDataFrame(_WeightedObject, pandas.DataFrame):
    """Weighted version of pandas.DataFrame."""

    def mean(self, axis=0, skipna=True):
        """Weighted mean of the sampled distribution."""
        if axis == 0:
            null = self.isnull() & skipna
            mean = np.average(masked_array(self, null),
                              weights=self.weights, axis=0)
            return pandas.Series(mean, index=self.columns)
        else:
            return super().mean(axis=axis, skipna=skipna)

    def var(self, axis=0, skipna=True):
        """Weighted variance of the sampled distribution."""
        if axis == 0:
            null = self.isnull() & skipna
            mean = self.mean(skipna=skipna).values
            var = np.average((masked_array(self, null)-mean)**2,
                             weights=self.weights, axis=0)
            return pandas.Series(var, index=self.columns)
        else:
            return super().var(axis=axis, skipna=skipna)

    def cov(self, skipna=True):
        """Weighted covariance of the sampled distribution."""
        null = self.isnull() & skipna
        mean = self.mean(skipna=skipna).values
        x = masked_array(self - mean, null)
        cov = np.ma.dot(self.weights * x.T, x) / self.weights.sum().T
        return pandas.DataFrame(cov, index=self.columns, columns=self.columns)

    def quantile(self, q=0.5, axis=0):
        """Weighted quantile of the sampled distribution."""
        if axis == 0:
            data = np.array([c.quantile(q) for _, c in self.iteritems()])
            if np.isscalar(q):
                return pandas.Series(data, index=self.columns)
            else:
                return pandas.DataFrame(data.T, columns=self.columns, index=q)
        else:
            return super().quantile(q=q, axis=axis)

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
        data = np.repeat(self.values, i, axis=0)
        index = self.index.repeat(i)
        df = pandas.DataFrame(data=data, index=index, columns=self.columns)
        df.index = df.index.get_level_values('#')
        return df

    @property
    def _constructor_sliced(self):
        return WeightedSeries

    @property
    def _constructor(self):
        return WeightedDataFrame
