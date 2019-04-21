"""Pandas DataFrame and Series with weighted samples."""

import numpy
import pandas


class WeightedSeries(pandas.Series):
    """Weighted version of pandas.Series."""

    _metadata = ['w']

    @property
    def _constructor(self):
        return WeightedSeries

    def __init__(self, *args, **kwargs):
        self.w = kwargs.pop('w', None)
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


class WeightedDataFrame(pandas.DataFrame):
    """Weighted version of pandas.DataFrame."""

    _metadata = ['w']

    @property
    def _constructor_sliced(self):
        def __constructor_sliced(*args, **kwargs):
            return WeightedSeries(*args, w=self.w, **kwargs)
        return __constructor_sliced

    @property
    def _constructor(self):
        return WeightedDataFrame

    def __init__(self, *args, **kwargs):
        self.w = kwargs.pop('w', None)
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
