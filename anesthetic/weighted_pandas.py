"""Pandas DataFrame and Series with weighted samples."""

import numpy as np
import pandas
from warnings import warn
from anesthetic.utils import compress_weights, channel_capacity, quantile


class _WeightedObject(object):
    @property
    def weight(self):
        """Sample weights."""
        if self._weight is None:
            return pandas.Series(index=self.index, data=1.)
        else:
            return self._weight[self.index]

    @property
    def _rand(self):
        """Random number for consistent compression."""
        return self._rand_[self.index]

    def _construct_weights(self, weight):
        if weight is not None:
            self._weight = pandas.Series(index=self.index, data=weight)
        else:
            self._weight = None
        rand = np.random.rand(len(self))
        self._rand_ = pandas.Series(index=self.index, data=rand)

    def std(self):
        """Weighted standard deviation of the sampled distribution."""
        return np.sqrt(self.var())

    def median(self):
        """Weighted median of the sampled distribution."""
        return self.quantile()

    def neff(self):
        """Effective number of samples."""
        return channel_capacity(self.weight)


class WeightedSeries(_WeightedObject, pandas.Series):
    """Weighted version of pandas.Series."""

    def __init__(self, *args, **kwargs):
        if 'w' in kwargs:
            warn("'w' as a kwarg will be deprecated in the future. "
                 "Please use 'weight'", FutureWarning)
        weight = kwargs.pop('w', None)
        weight = kwargs.pop('weight', weight)
        super(WeightedSeries, self).__init__(*args, **kwargs)
        self._construct_weights(weight)

    def mean(self):
        """Weighted mean of the sampled distribution."""
        return np.average(self, weights=self.weight)

    def var(self):
        """Weighted variance of the sampled distribution."""
        return np.average((self-self.mean())**2, weights=self.weight)

    def quantile(self, q=0.5):
        """Weighted quantile of the sampled distribution."""
        return quantile(self.values, q, self.weight.values)

    def hist(self, *args, **kwargs):
        """Weighted histogram of the sampled distribution."""
        return super(WeightedSeries, self).hist(weights=self.weight,
                                                *args, **kwargs)

    def compress(self, nsamples=None):
        """Reduce the number of samples by discarding low-weights.

        Parameters
        ----------
        neff: int, optional
            effective number of samples after compression. If not supplied,
            then reduce to the channel capacity (theoretical optimum
            compression). If <=0, then compress so that all weights are unity.

        """
        i = compress_weights(self.weight, self._rand, nsamples)
        return self.repeat(i)

    _metadata = ['_weight', '_rand_']

    @property
    def _constructor(self):
        return WeightedSeries

    @property
    def _constructor_expanddim(self):
        def __constructor_expanddim(*args, **kwargs):
            frame = WeightedDataFrame(*args, weight=self._weight, **kwargs)
            frame._rand_ = self._rand_
            return frame
        return __constructor_expanddim


class WeightedDataFrame(_WeightedObject, pandas.DataFrame):
    """Weighted version of pandas.DataFrame."""

    def __init__(self, *args, **kwargs):
        if 'w' in kwargs:
            warn("'w' as a kwarg will be deprecated in the future. "
                 "Please use 'weight'", FutureWarning)
        weight = kwargs.pop('w', None)
        weight = kwargs.pop('weight', weight)
        super(WeightedDataFrame, self).__init__(*args, **kwargs)
        self._construct_weights(weight)

    def mean(self):
        """Weighted mean of the sampled distribution."""
        return pandas.Series(np.average(self, weights=self.weight, axis=0),
                             index=self.columns)

    def var(self):
        """Weighted variance of the sampled distribution."""
        return pandas.Series(np.average((self-self.mean())**2,
                                        weights=self.weight, axis=0),
                             index=self.columns)

    def cov(self):
        """Weighted covariance of the sampled distribution."""
        return pandas.DataFrame(np.cov(self.T, aweights=self.weight),
                                index=self.columns, columns=self.columns)

    def quantile(self, q=0.5):
        """Weighted quantile of the sampled distribution."""
        data = np.array([c.quantile(q) for _, c in self.iteritems()])
        if np.isscalar(q):
            return pandas.Series(data, index=self.columns)
        else:
            return pandas.DataFrame(data.T, columns=self.columns, index=q)

    def hist(self, *args, **kwargs):
        """Weighted histogram of the sampled distribution."""
        return super(WeightedDataFrame, self).hist(weights=self.weight,
                                                   *args, **kwargs)

    def compress(self, nsamples=None):
        """Reduce the number of samples by discarding low-weights.

        Parameters
        ----------
        neff: int, optional
            effective number of samples after compression. If not supplied,
            then reduce to the channel capacity (theoretical optimum
            compression). If <=0, then compress so that all weights are unity.

        """
        i = compress_weights(self.weight, self._rand, nsamples)
        data = np.repeat(self.values, i, axis=0)
        index = np.repeat(self.index.values, i)
        df = pandas.DataFrame(data=data, index=index, columns=self.columns)
        if 'weight' in self:
            return df.drop(columns='weight')
        else:
            return df

    _metadata = ['_weight', '_rand_']

    @property
    def _constructor_sliced(self):
        def __constructor_sliced(*args, **kwargs):
            series = WeightedSeries(*args, weight=self._weight, **kwargs)
            series._rand_ = self._rand_
            return series
        return __constructor_sliced

    @property
    def _constructor(self):
        return WeightedDataFrame
