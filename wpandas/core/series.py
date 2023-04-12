"""
Weighted data structure for 1-dimensional cross-sectional and time series data
"""
import numpy as np
from numpy.ma import masked_array
from pandas import Series
from wpandas.core.util._code_transforms import adjust_weighted_docstrings
from wpandas.core.base import _WeightedObject

from wpandas.core.util.weights import compress_weights, quantile
from pandas.core.dtypes.missing import notna


class WeightedSeries(_WeightedObject, Series):
    """Weighted version of :class:`pandas.Series`."""

    def mean(self, skipna=True):  # noqa: D102
        if self.get_weights().sum() == 0:
            return np.nan
        null = self.isnull() & skipna
        return np.average(masked_array(self, null), weights=self.get_weights())

    def std(self, *args, **kwargs):  # noqa: D102
        return np.sqrt(self.var(*args, **kwargs))

    def kurtosis(self, *args, **kwargs):  # noqa: D102
        return self.kurt(*args, **kwargs)

    def median(self, *args, **kwargs):  # noqa: D102
        if self.get_weights().sum() == 0:
            return np.nan
        return self.quantile(*args, **kwargs)

    def var(self, skipna=True):  # noqa: D102
        if self.get_weights().sum() == 0:
            return np.nan
        null = self.isnull() & skipna
        mean = self.mean(skipna=skipna)
        if np.isnan(mean):
            return mean
        return np.average(masked_array((self-mean)**2, null),
                          weights=self.get_weights())

    def cov(self, other, *args, **kwargs):  # noqa: D102

        this, other = self.align(other, join="inner", copy=False)
        if len(this) == 0:
            return np.nan

        weights = self.index.to_frame()['weights']
        weights, _ = weights.align(other, join="inner", copy=False)

        valid = notna(this) & notna(other)
        if not valid.all():
            this = this[valid]
            other = other[valid]
            weights = weights[valid]

        return np.cov(this, other, aweights=weights)[0, 1]

    def corr(self, other, *args, **kwargs):  # noqa: D102
        norm = self.std(skipna=True)*other.std(skipna=True)
        return self.cov(other, *args, **kwargs)/norm

    def kurt(self, skipna=True):  # noqa: D102
        if self.get_weights().sum() == 0:
            return np.nan
        null = self.isnull() & skipna
        mean = self.mean(skipna=skipna)
        std = self.std(skipna=skipna)
        if np.isnan(mean) or np.isnan(std):
            return np.nan
        return np.average(masked_array(((self-mean)/std)**4, null),
                          weights=self.get_weights())

    def skew(self, skipna=True):  # noqa: D102
        if self.get_weights().sum() == 0:
            return np.nan
        null = self.isnull() & skipna
        mean = self.mean(skipna=skipna)
        std = self.std(skipna=skipna)
        if np.isnan(mean) or np.isnan(std):
            return np.nan
        return np.average(masked_array(((self-mean)/std)**3, null),
                          weights=self.get_weights())

    def sem(self, skipna=True):  # noqa: D102
        return np.sqrt(self.var(skipna=skipna)/self.neff())

    def quantile(self, q=0.5, interpolation='linear'):  # noqa: D102
        if self.get_weights().sum() == 0:
            return np.nan
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
        from wpandas.core.frame import WeightedDataFrame
        return WeightedDataFrame

    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        observed=False,
        dropna=True,
    ):  # pragma: no cover  # noqa: D102
        from wpandas.core.groupby import WeightedSeriesGroupBy

        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        if not as_index:
            raise TypeError("as_index=False only valid with DataFrame")
        axis = self._get_axis_number(axis)

        return WeightedSeriesGroupBy(
            obj=self,
            keys=by,
            axis=axis,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )


adjust_weighted_docstrings(WeightedSeries)
