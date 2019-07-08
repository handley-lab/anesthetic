"""Pandas DataFrame and Series with weighted samples."""

import numpy
import pandas
from anesthetic.utils import (compress_weights, channel_capacity, quantile)


class WeightedSeries(pandas.Series):
    """Weighted version of pandas.Series."""

    _metadata = ['w', 'u']

    @property
    def _constructor(self):
        return WeightedSeries

    def __init__(self, *args, **kwargs):
        self.w = kwargs.pop('w', None)
        self.u = kwargs.pop('u', None)
        super(WeightedSeries, self).__init__(*args, **kwargs)

    def mean(self):
        """Weighted mean of the sampled distribution."""
        return numpy.average(self, weights=self.w)

    def std(self):
        """Weighted standard deviation of the sampled distribution."""
        return numpy.sqrt(self.var())

    def var(self):
        """Weighted variance of the sampled distribution."""
        return numpy.average((self-self.mean())**2, weights=self.w)

    def median(self):
        """Weighted median of the sampled distribution."""
        return float(self.quantile())

    def quantile(self, q=0.5):
        """Weighted quantile of the sampled distribution."""
        return quantile(self.values, q, self.w)

    def hist(self, *args, **kwargs):
        """Weighted histogram of the sampled distribution."""
        return super(WeightedSeries, self).hist(weights=self.w, *args, **kwargs)

    def neff(self):
        """Effective number of samples."""
        if self.w is None:
            return len(self)
        else:
            return channel_capacity(self.w)

    def compress(self, nsamples=None):
        """Reduce the number of samples by discarding low-weights.

        Parameters
        ----------
        neff: int, optional
            effective number of samples after compression. If not supplied,
            then reduce to the channel capacity (theoretical optimum
            compression). If <=0, then compress so that all weights are unity.

        """
        i = compress_weights(self.w, self.u, nsamples)
        return self.repeat(i)


class WeightedDataFrame(pandas.DataFrame):
    """Weighted version of pandas.DataFrame."""

    _metadata = ['w', 'u']

    @property
    def _constructor_sliced(self):
        def __constructor_sliced(*args, **kwargs):
            return WeightedSeries(*args, w=self.w, u=self.u, **kwargs)
        return __constructor_sliced

    @property
    def _constructor(self):
        return WeightedDataFrame

    def __init__(self, *args, **kwargs):
        self.w = kwargs.pop('w', None)
        self.u = kwargs.pop('u', None)
        super(WeightedDataFrame, self).__init__(*args, **kwargs)

    def mean(self):
        """Weighted mean of the sampled distribution."""
        return pandas.Series(numpy.average(self, weights=self.w, axis=0),
                             index=self.columns)

    def std(self):
        """Weighted standard deviation of the sampled distribution."""
        return numpy.sqrt(self.var())

    def var(self):
        """Weighted variance of the sampled distribution."""
        return pandas.Series(numpy.average((self-self.mean())**2,
                                           weights=self.w, axis=0),
                             index=self.columns)

    def cov(self):
        """Weighted covariance of the sampled distribution."""
        return pandas.DataFrame(numpy.cov(self.T, aweights=self.w),
                                index=self.columns, columns=self.columns)

    def median(self):
        """Weighted median of the sampled distribution."""
        return self.quantile()

    def quantile(self, q=0.5):
        """Weighted quantile of the sampled distribution."""
        data = numpy.array([c.quantile(q) for _, c in self.iteritems()])
        if numpy.isscalar(q):
            return pandas.Series(data, index=self.columns)
        else:
            return pandas.DataFrame(data.T, columns=self.columns, index=q)

    def hist(self, *args, **kwargs):
        """Weighted histogram of the sampled distribution."""
        return super(WeightedDataFrame, self).hist(weights=self.w, *args, **kwargs)

    def neff(self):
        """Effective number of samples."""
        if self.w is None:
            return len(self)
        else:
            return channel_capacity(self.w)

    def compress(self, nsamples=None):
        """Reduce the number of samples by discarding low-weights.

        Parameters
        ----------
        neff: int, optional
            effective number of samples after compression. If not supplied,
            then reduce to the channel capacity (theoretical optimum
            compression). If <=0, then compress so that all weights are unity.

        """
        i = compress_weights(self.w, self.u, nsamples)
        data = numpy.repeat(self.values, i, axis=0)
        index = numpy.repeat(self.index.values, i)
        return pandas.DataFrame(data=data, index=index, columns=self.columns)
