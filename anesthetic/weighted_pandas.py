"""Pandas DataFrame and Series with weighted samples."""

import numpy as np
import pandas
from anesthetic.utils import (compress_weights, channel_capacity, quantile,
                              temporary_seed)


class _WeightedObject(object):
    @property
    def weights(self):
        """Sample weights."""
        if self.index.nlevels == 1:
            return pandas.Series(index=self.index, data=1.)
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

    def std(self):
        """Weighted standard deviation of the sampled distribution."""
        return np.sqrt(self.var())

    def kurtosis(self):
        """Weighted standard deviation of the sampled distribution."""
        return self.kurt()

    def median(self):
        """Weighted median of the sampled distribution."""
        return self.quantile()

    def neff(self):
        """Effective number of samples."""
        return channel_capacity(self.weights)

    def sem(self):
        """Weighted standard error of the mean."""
        return np.sqrt(self.var()/self.neff())


class WeightedSeries(_WeightedObject, pandas.Series):
    """Weighted version of pandas.Series."""

    def __init__(self, *args, **kwargs):
        weights = kwargs.pop('weights', None)
        super(WeightedSeries, self).__init__(*args, **kwargs)
        self.weights = weights

    def mean(self):
        """Weighted mean of the sampled distribution."""
        nonzero = self.weights != 0
        return np.average(self[nonzero], weights=self.weights[nonzero])

    def var(self):
        """Weighted variance of the sampled distribution."""
        nonzero = self.weights != 0
        return np.average((self[nonzero]-self.mean())**2,
                          weights=self.weights[nonzero])

    def cov(self, other):
        """Weighted covariance with another Series."""
        nonzero = self.weights != 0
        return np.cov(self[nonzero], other[nonzero],
                      aweights=self.weights[nonzero])[0, 1]

    def corr(self, other):
        """Weighted pearson correlation with another Series."""
        nonzero = self.weights != 0
        cov = np.cov(self[nonzero], other[nonzero],
                     aweights=self.weights[nonzero])
        return cov[0, 1]/np.sqrt(cov[0, 0])/np.sqrt(cov[1, 1])

    def kurt(self):
        """Weighted kurtosis of the sampled distribution."""
        nonzero = self.weights != 0
        return np.average(((self[nonzero]-self.mean())/self.std())**4,
                          weights=self.weights[nonzero])

    def skew(self):
        """Weighted skewness of the sampled distribution."""
        nonzero = self.weights != 0
        return np.average(((self[nonzero]-self.mean())/self.std())**3,
                          weights=self.weights[nonzero])

    def mad(self):
        """Weighted mean absolute deviation of the sampled distribution."""
        nonzero = self.weights != 0
        return np.average(abs(self[nonzero]-self.mean()),
                          weights=self.weights[nonzero])

    def quantile(self, q=0.5):
        """Weighted quantile of the sampled distribution."""
        return quantile(self.values, q, self.weights)

    def hist(self, *args, **kwargs):
        """Weighted histogram of the sampled distribution."""
        return super(WeightedSeries, self).hist(weights=self.weights,
                                                *args, **kwargs)

    def sample(self, *args, **kwargs):
        """Weighted sample."""
        return super(WeightedSeries, self).sample(weights=self.weights,
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

    def __init__(self, *args, **kwargs):
        weights = kwargs.pop('weights', None)
        super(WeightedDataFrame, self).__init__(*args, **kwargs)
        self.weights = weights

    def mean(self):
        """Weighted mean of the sampled distribution."""
        nonzero = self.weights != 0
        mean = np.average(self[nonzero], weights=self.weights[nonzero], axis=0)
        return pandas.Series(mean, index=self.columns)

    def var(self):
        """Weighted variance of the sampled distribution."""
        nonzero = self.weights != 0
        var = np.average((self[nonzero]-self.mean())**2,
                         weights=self.weights[nonzero], axis=0)
        return pandas.Series(var, index=self.columns)

    def cov(self):
        """Weighted covariance of the sampled distribution."""
        nonzero = self.weights != 0
        cov = np.cov(self[nonzero].T, aweights=self.weights[nonzero])
        return pandas.DataFrame(cov, index=self.columns, columns=self.columns)

    def corr(self):
        """Weighted pearson correlation matrix of the sampled distribution."""
        cov = self.cov()
        diag = np.sqrt(np.diag(cov))
        return cov.divide(diag, axis=1).divide(diag, axis=0)

    def corrwith(self, other, drop=False):
        """Weighted pearson correlation matrix of the sampled distribution."""
        this = self._get_numeric_data()

        if isinstance(other, pandas.Series):
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
                correl = correl.append(pandas.Series([np.nan] * len(idx_diff),
                                       index=idx_diff))

        return correl

    def kurt(self):
        """Weighted kurtosis of the sampled distribution."""
        nonzero = self.weights != 0
        kurt = np.average(((self[nonzero]-self.mean())/self.std())**4,
                          weights=self.weights[nonzero], axis=0)
        return pandas.Series(kurt, index=self.columns)

    def skew(self):
        """Weighted skewness of the sampled distribution."""
        nonzero = self.weights != 0
        skew = np.average(((self[nonzero]-self.mean())/self.std())**3,
                          weights=self.weights[nonzero], axis=0)
        return pandas.Series(skew, index=self.columns)

    def mad(self):
        """Weighted mean absolute deviation of the sampled distribution."""
        nonzero = self.weights != 0
        kurt = np.average(abs(self[nonzero]-self.mean()),
                          weights=self.weights[nonzero], axis=0)
        return pandas.Series(kurt, index=self.columns)

    def quantile(self, q=0.5):
        """Weighted quantile of the sampled distribution."""
        data = np.array([c.quantile(q) for _, c in self.iteritems()])
        if np.isscalar(q):
            return pandas.Series(data, index=self.columns)
        else:
            return pandas.DataFrame(data.T, columns=self.columns, index=q)

    def hist(self, *args, **kwargs):
        """Weighted histogram of the sampled distribution."""
        return super(WeightedDataFrame, self).hist(weights=self.weights,
                                                   *args, **kwargs)

    def sample(self, *args, **kwargs):
        """Weighted sample."""
        return super(WeightedDataFrame, self).sample(weights=self.weights,
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
