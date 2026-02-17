"""Pandas DataFrame and Series with weighted samples."""

import warnings
from inspect import signature
import numpy as np
from pandas import Series, DataFrame, concat, MultiIndex
from pandas.core.groupby import GroupBy, SeriesGroupBy, DataFrameGroupBy, ops
from pandas._libs import lib
from pandas._libs.lib import no_default
from pandas.util._exceptions import find_stack_level
from pandas.util import hash_pandas_object
from anesthetic.utils import (compress_weights, neff, quantile,
                              temporary_seed, adjust_docstrings,
                              var_unbiased, cov_unbiased, skew_unbiased,
                              kurt_unbiased, credibility_interval)
from pandas.core.accessor import CachedAccessor
from anesthetic.plotting import PlotAccessor
import pandas as pd


def read_csv(filename, *args, **kwargs):
    """Read a CSV file into a ``WeightedDataFrame``."""
    df = pd.read_csv(filename, index_col=[0, 1], header=[0, 1],
                     *args, **kwargs)
    wdf = WeightedDataFrame(df)
    if wdf.isweighted(0) and wdf.isweighted(1):
        wdf.set_weights(wdf.get_weights(axis=1).astype(float),
                        axis=1, inplace=True)
        return wdf
    df = pd.read_csv(filename, index_col=[0, 1], *args, **kwargs)
    wdf = WeightedDataFrame(df)
    if wdf.isweighted(0):
        return wdf
    df = pd.read_csv(filename, index_col=0, header=[0, 1], *args, **kwargs)
    wdf = WeightedDataFrame(df)
    if wdf.isweighted(1):
        wdf.set_weights(wdf.get_weights(axis=1).astype(float),
                        axis=1, inplace=True)
        return wdf
    df = pd.read_csv(filename, index_col=0, *args, **kwargs)
    return WeightedDataFrame(df)


class WeightedGroupBy(GroupBy):
    """Weighted version of ``pandas.core.groupby.GroupBy``."""

    _grouper: ops.BaseGrouper

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _add_weights(self, name, *args, **kwargs):
        result = self.agg(lambda df: getattr(self.obj._constructor(df), name)
                          (*args, **kwargs)).set_weights(self.get_weights())
        return result.__finalize__(self.obj, method="groupby")

    def mean(self, **kwargs):  # noqa: D102
        return self._add_weights("mean", **kwargs)

    def std(self, **kwargs):  # noqa: D102
        return self._add_weights("std", **kwargs)

    def median(self, **kwargs):  # noqa: D102
        return self._add_weights("median", **kwargs)

    def var(self, **kwargs):  # noqa: D102
        return self._add_weights("var", **kwargs)

    def kurt(self, **kwargs):  # noqa: D102
        return self._add_weights("kurt", **kwargs)

    def kurtosis(self, **kwargs):  # noqa: D102
        return self._add_weights("kurtosis", **kwargs)

    def sem(self, **kwargs):  # noqa: D102
        return self._add_weights("sem", **kwargs)

    def skew(self, **kwargs):  # noqa: D102
        return self._add_weights("skew", **kwargs)

    def quantile(self, *args, **kwargs):  # noqa: D102
        return self._add_weights("quantile", *args, **kwargs)

    def get_weights(self):
        """Return the weights of the grouped samples."""
        return self.agg(lambda df: df.get_weights().sum())

    def _op_via_apply(self, name, *args, **kwargs):
        result = super()._op_via_apply(name, *args, **kwargs)
        try:
            index = result.index.get_level_values(self.keys)
            weights = self.get_weights()[index]
        except KeyError:
            weights = self.get_weights()
        return result.set_weights(weights, level=1)


class WeightedSeriesGroupBy(WeightedGroupBy, SeriesGroupBy):
    """Weighted version of ``pandas.core.groupby.SeriesGroupBy``."""

    def sample(self, *args, **kwargs):  # noqa: D102
        return super().sample(weights=self.obj.get_weights(), *args, **kwargs)

    def cov(self, *args, **kwargs):  # noqa: D102
        return self._op_via_apply("cov", *args, **kwargs)


class WeightedDataFrameGroupBy(WeightedGroupBy, DataFrameGroupBy):
    """Weighted version of ``pandas.core.groupby.DataFrameGroupBy``."""

    def get_weights(self):
        """Return the weights of the grouped samples."""
        return super().get_weights().min(axis=1)

    def _gotitem(self, key, ndim: int, subset=None):  # pragma: no cover
        if ndim == 2:
            if subset is None:
                subset = self.obj
            return WeightedDataFrameGroupBy(
                subset,
                self._grouper,
                level=self.level,
                grouper=self._grouper,
                exclusions=self.exclusions,
                selection=key,
                as_index=self.as_index,
                sort=self.sort,
                group_keys=self.group_keys,
                observed=self.observed,
                dropna=self.dropna,
            )
        elif ndim == 1:
            if subset is None:
                subset = self.obj[key]
            return WeightedSeriesGroupBy(
                subset,
                level=self.level,
                grouper=self._grouper,
                selection=key,
                sort=self.sort,
                group_keys=self.group_keys,
                observed=self.observed,
                dropna=self.dropna,
            )

        raise AssertionError("invalid ndim for _gotitem")

    def sample(self, *args, **kwargs):  # noqa: D102
        return super().sample(weights=self.obj.get_weights(), *args, **kwargs)

    def cov(self, *args, **kwargs):  # noqa: D102
        return self._op_via_apply("cov", *args, **kwargs)


class _WeightedObject(object):
    """Common methods for `WeightedSeries` and `WeightedDataFrame`.

    :meta public:
    """

    def __init__(self, *args, **kwargs):
        weights = kwargs.pop('weights', None)
        super().__init__(*args, **kwargs)
        if weights is not None:
            self.set_weights(weights, inplace=True)

    plot = CachedAccessor("plot", PlotAccessor)
    """:meta private:"""

    def isweighted(self, axis=0):
        """Determine if weights are actually present."""
        return 'weights' in self._get_axis(axis).names

    def get_weights(self, axis=0):
        """Retrieve sample weights from an axis."""
        if self.isweighted(axis):
            return self._get_axis(axis).get_level_values('weights').to_numpy()
        else:
            return np.ones_like(self._get_axis(axis), dtype=int)

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
            result = result.set_axis(index, axis=axis)

        if inplace:
            self._update_inplace(result)
        else:
            return result.__finalize__(self, "set_weights")

    def _rand(self, axis=0):
        """Random number for consistent compression."""
        seed = hash_pandas_object(self._get_axis(axis)).sum() % 2**32
        with temporary_seed(int(seed)):
            return np.random.rand(self.shape[axis])

    def _weighted_stat(self, func, axis=0, skipna=True, **kwargs):
        """Compute weighted statistics using common pattern."""
        if not self.isweighted(axis):
            # Get the calling method name automatically
            method_name = func.__name__

            # Check if the method exists in pandas DataFrame
            if hasattr(super(), method_name):
                return getattr(super(), method_name)(axis=axis, skipna=skipna,
                                                     **kwargs)

        if self.get_weights(axis).sum() == 0:
            return self._constructor_sliced(np.nan,
                                            index=self._get_axis(1-axis))

        na = self.isna() & skipna
        weights = np.broadcast_to(
            self.get_weights(axis)[..., None] if axis == 0 else
            self.get_weights(axis)[None, ...],
            self.shape
        )
        if skipna:
            weights = np.ma.array(weights, mask=na)
        result = np.ma.filled(
            func(self, na=na, w=weights, axis=axis, skipna=skipna, **kwargs),
            np.nan
        )
        return self._constructor_sliced(result, index=self._get_axis(1-axis))

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

    def neff(self, axis=0, beta=1):
        """Effective number of samples."""
        if self.isweighted(axis):
            return neff(self.get_weights(axis), beta=beta)
        else:
            return self.shape[axis]


class WeightedSeries(_WeightedObject, Series):
    """Weighted version of :class:`pandas.Series`."""

    def mean(self, skipna=True):  # noqa: D102
        na = self.isna() & skipna
        weights = self.get_weights()
        if skipna:
            weights = np.ma.array(self.get_weights(), mask=na)
        if weights.sum() == 0 or skipna and na.all():
            return np.nan
        return np.average(np.ma.array(self, mask=na), weights=weights)

    def std(self, skipna=True, **kwargs):  # noqa: D102
        return np.sqrt(self.var(skipna=skipna, **kwargs))

    def kurtosis(self, **kwargs):  # noqa: D102
        return self.kurt(**kwargs)

    def median(self, **kwargs):  # noqa: D102
        if self.get_weights().sum() == 0:
            return np.nan
        return self.quantile(**kwargs)

    def var(self, skipna=True, **kwargs):  # noqa: D102
        na = self.isna() & skipna
        w = np.ma.array(self.get_weights(), mask=na)
        if na.all() or self.isna().any() and not skipna or w.sum() == 0:
            return np.float64(np.nan)
        return var_unbiased(np.ma.array(self, mask=na), w, **kwargs)

    def cov(self, other, ddof=1, **kwargs):  # noqa: D102
        w = self.get_weights()
        x, y = self.align(other, join="inner")
        if len(x) == 0:
            return np.nan
        valid = x.notna() & y.notna()
        x = x[valid]
        y = y[valid]
        w = w[valid]
        if len(x) == 0 or w.sum() == 0:
            return np.nan
        X = np.column_stack((x.to_numpy(dtype=float), y.to_numpy(dtype=float)))
        return cov_unbiased(X, w, ddof=ddof)[0, 1]

    def corr(self, other, **kwargs):  # noqa: D102
        if not self.isweighted():
            return super().corr(other, **kwargs)
        if self.isna().all():
            return np.nan
        norm = (self.std(skipna=True, ddof=1) *
                other.std(skipna=True, ddof=1))
        if norm == 0:
            return np.float64(np.nan)
        return self.cov(other, ddof=1) / norm

    def kurt(self, skipna=True, **kwargs):  # noqa: D102
        if self.isna().all() or self.isna().any() and not skipna:
            return np.nan if skipna or self.size == 1 else np.float64(np.nan)
        na = self.isna() & skipna
        w = np.ma.array(self.get_weights(), mask=na)
        return kurt_unbiased(np.ma.array(self, mask=na), w)

    def skew(self, skipna=True, **kwargs):  # noqa: D102
        if self.isna().all() or self.isna().any() and not skipna:
            return np.nan if skipna or self.size == 1 else np.float64(np.nan)
        na = self.isna() & skipna
        w = np.ma.array(self.get_weights(), mask=na)
        return skew_unbiased(np.ma.array(self, mask=na), w)

    def sem(self, skipna=True, ddof=1, **kwargs):  # noqa: D102
        na = self.isna() & skipna
        w = np.ma.array(self.get_weights(), mask=na)
        V1 = w.sum()
        if np.issubdtype(w.dtype, np.integer) and V1 > 1:
            # frequency weights
            n = np.ma.filled(V1, np.nan)
        else:
            # reliability weights
            n = np.ma.filled(V1**2 / (w**2).sum(), np.nan)
        return np.sqrt(self.var(skipna=skipna, ddof=ddof, **kwargs) / n)

    def quantile(self, q=0.5, interpolation='linear'):  # noqa: D102
        if self.get_weights().sum() == 0:
            return np.nan
        return quantile(self.to_numpy(), q, self.get_weights(), interpolation)

    def compress(self, ncompress=True):
        """Reduce the number of samples by discarding low-weights.

        Parameters
        ----------
        ncompress : int, float, str, default=True
            Degree of compression.

            * If ``True`` (default): reduce to the channel capacity
              (theoretical optimum compression), equivalent to
              ``ncompress='entropy'``.
            * If ``> 0``: desired number of samples after compression.
            * If ``<= 0``: compress so that all remaining weights are unity.
            * If ``str``: determine number from the Huggins-Roy family of
              effective samples in :func:`anesthetic.utils.neff`
              with ``beta=ncompress``.

        """
        i = compress_weights(self.get_weights(), self._rand(), ncompress)
        return self.repeat(i)

    def sample(self, *args, **kwargs):  # noqa: D102
        return super().sample(weights=self.get_weights(), *args, **kwargs)

    def credibility_interval(self, level=0.68, method="iso-pdf",
                             return_covariance=False, nsamples=12):
        """Compute the credibility interval of the weighted samples.

        Based on linear interpolation of the cumulative density function, thus
        expect discretisation errors on the scale of distances between samples.

        https://github.com/Stefan-Heimersheim/fastCI#readme

        Parameters
        ----------
        level : float, default=0.68
            Credibility level (probability, <1).
        method : str, default='iso-pdf'
            Which definition of interval to use:

            * ``'iso-pdf'``: Calculate iso probability density interval with
              the same probability density at each end. Also known as
              waterline-interval or highest average posterior density interval.
              This is only accurate if the distribution is sufficiently
              uni-modal.
            * ``'lower-limit'``/``'upper-limit'``: Lower/upper limit. One-sided
              limits for which ``level`` fraction of the (equally weighted)
              samples lie above/below the limit.
            * ``'equal-tailed'``: Equal-tailed interval with the same fraction
              of (equally weighted) samples below and above the interval
              region.

        return_covariance: bool, default=False
            Return the covariance of the sampled limits, in addition to the
            mean
        nsamples : int, default=12
            Number of CDF samples to improve `mean` and `std` estimate.

        Returns
        -------
        limit(s) : float, array, or tuple of floats or arrays
            Returns the credibility interval boundaries of the Series.
            By default, returns the mean over ``nsamples`` samples, which is
            either two numbers (``method='iso-pdf'``/``'equal-tailed'``) or
            one number (``method='lower-limit'``/``'upper-limit'``). If
            ``return_covariance=True``, returns a tuple (mean(s), covariance)
            where covariance is the covariance over the sampled limits.
        """
        return credibility_interval(self, weights=self.get_weights(),
                                    level=level, method=method,
                                    return_covariance=return_covariance,
                                    nsamples=nsamples)

    @property
    def _constructor(self):
        return WeightedSeries

    @property
    def _constructor_expanddim(self):
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
        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        if not as_index:
            raise TypeError("as_index=False only valid with DataFrame")
        axis = self._get_axis_number(axis)

        return WeightedSeriesGroupBy(
            obj=self,
            keys=by,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )


class WeightedDataFrame(_WeightedObject, DataFrame):
    """Weighted version of :class:`pandas.DataFrame`."""

    def mean(self, axis=0, skipna=True, **kwargs):  # noqa: D102
        def mean(data, na, w, axis, skipna, **kwargs):
            if skipna:
                data = np.ma.array(data, mask=na)
            return np.average(data, weights=w, axis=axis)

        return self._weighted_stat(mean, axis, skipna, **kwargs)

    def std(self, axis=0, skipna=True, **kwargs):  # noqa: D102
        return np.sqrt(self.var(axis=axis, skipna=skipna, **kwargs))

    def kurtosis(self, **kwargs):  # noqa: D102
        return self.kurt(**kwargs)

    def median(self, **kwargs):  # noqa: D102
        return self.quantile(**kwargs)

    def var(self, axis=0, skipna=True, **kwargs):  # noqa: D102
        def var(data, na, w, axis, skipna, **kwargs):
            if skipna:
                data = np.ma.array(data, mask=na)
            return var_unbiased(data, w, axis=axis, **kwargs)
        return self._weighted_stat(var, axis, skipna, **kwargs)

    def cov(self, ddof=1, **kwargs):  # noqa: D102
        if kwargs:
            raise TypeError(f"WeightedDataFrame.cov() got unexpected keyword "
                            f"arguments {kwargs}")

        if not self.isweighted():
            return super().cov(ddof=ddof)

        cov = cov_unbiased(self, self.get_weights(), ddof=ddof)
        return self._constructor(cov, index=self.columns, columns=self.columns)

    def corr(self, **kwargs):  # noqa: D102
        if not self.isweighted():
            return super().corr(**kwargs)
        corr = cov_unbiased(self, self.get_weights(), ddof=1, return_corr=True)
        return self._constructor(corr, index=self.columns,
                                 columns=self.columns)

    def corrwith(self, other, axis=0, drop=False, **kwargs):  # noqa: D102
        axis = self._get_axis_number(axis)
        if not self.isweighted(axis):
            return super().corrwith(other, drop=drop, axis=axis, **kwargs)
        else:
            if isinstance(other, Series):
                answer = self.apply(lambda x: other.corr(x, **kwargs),
                                    axis=axis)
                return self._constructor_sliced(answer)

            left, right = self.align(other, join="inner")

            if axis == 1:
                left = left.T
                right = right.T

            weights = left.index.to_frame()['weights']
            weights, _ = weights.align(right, join="inner")

            # mask missing values
            left = left + right * 0
            right = right + left * 0

            # demeaned data
            ldem = left - left.mean()
            rdem = right - right.mean()

            ddof = kwargs.pop('ddof', 0)
            num = (ldem * rdem * weights.to_numpy()[:, None]).sum()
            dom = weights.sum() * left.std(ddof=ddof) * right.std(ddof=ddof)

            correl = num / dom

            if not drop:
                # Find non-matching labels along the given axis
                result_index = self._get_axis(1-axis).union(
                    other._get_axis(1-axis)
                )
                idx_diff = result_index.difference(correl.index)

                if len(idx_diff) > 0:
                    correl = concat([
                        correl,
                        Series([np.nan] * len(idx_diff), index=idx_diff)
                    ])

            return self._constructor_sliced(correl)

    def kurt(self, axis=0, skipna=True, **kwargs):  # noqa: D102
        def kurt(data, na, w, axis, skipna, **kwargs):
            if skipna:
                data = np.ma.array(data, mask=na)
            return kurt_unbiased(data, w, axis=axis)
        return self._weighted_stat(kurt, axis, skipna, **kwargs)

    def skew(self, axis=0, skipna=True, **kwargs):  # noqa: D102
        def skew(data, na, w, axis, skipna, **kwargs):
            if skipna:
                data = np.ma.array(data, mask=na)
            return skew_unbiased(data, w, axis=axis)
        return self._weighted_stat(skew, axis, skipna, **kwargs)

    def sem(self, axis=0, skipna=True, **kwargs):  # noqa: D102
        def sem(data, na, w, axis, skipna, **kwargs):
            V1 = w.sum(axis=axis)
            if np.issubdtype(w.dtype, np.integer) and np.all(V1 > 1):
                # frequency weights
                n = V1
            else:
                # reliability weights
                n = V1**2 / (w**2).sum(axis=axis)
            return np.sqrt(self.var(axis=axis, skipna=skipna, **kwargs) / n)
        return self._weighted_stat(sem, axis, skipna, **kwargs)

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
        ncompress : int, float, str, default=True
            Degree of compression.

            * If ``True`` (default): reduce to the channel capacity
              (theoretical optimum compression), equivalent to
              ``ncompress='entropy'``.
            * If ``> 0``: desired number of samples after compression.
            * If ``<= 0``: compress so that all remaining weights are unity.
            * If ``str``: determine number from the Huggins-Roy family of
              effective samples in :func:`anesthetic.utils.neff`
              with ``beta=ncompress``.

        """
        if (not self.isweighted(axis) and isinstance(ncompress, (bool, str))
                or ncompress is False):
            return self
        i = compress_weights(self.get_weights(axis), self._rand(axis),
                             ncompress)
        data = np.repeat(self.to_numpy(), i, axis=axis)
        i = self.drop_weights(axis)._get_axis(axis).repeat(i)
        df = self._constructor(data=data)
        df = df.set_axis(i, axis=axis)
        df = df.set_axis(self._get_axis(1-axis), axis=1-axis)
        return df

    def sample(self, *args, **kwargs):  # noqa: D102
        sig = signature(DataFrame.sample)
        axis = sig.bind(self, *args, **kwargs).arguments.get('axis', 0)
        if self.isweighted(axis):
            return super().sample(weights=self.get_weights(axis),
                                  *args, **kwargs)
        else:
            return super().sample(*args, **kwargs)

    def credibility_interval(self, level=0.68, method="iso-pdf",
                             return_covariance=False, nsamples=12):
        """Compute the credibility interval of the weighted samples.

        Based on linear interpolation of the cumulative density function, thus
        expect discretisation errors on the scale of distances between samples.

        https://github.com/Stefan-Heimersheim/fastCI#readme

        Parameters
        ----------
        level : float, default=0.68
            Credibility level (probability, <1).
        method : str, default='iso-pdf'
            Which definition of interval to use:

            * ``'iso-pdf'``: Calculate iso probability density interval with
              the same probability density at each end. Also known as
              waterline-interval or highest average posterior density interval.
              This is only accurate if the distribution is sufficiently
              uni-modal.
            * ``'lower-limit'``/``'upper-limit'``: Lower/upper limit. One-sided
              limits for which ``level`` fraction of the (equally weighted)
              samples lie above/below the limit.
            * ``'equal-tailed'``: Equal-tailed interval with the same fraction
              of (equally weighted) samples below and above the interval
              region.

        return_covariance: bool, default=False
            Return the covariance of the sampled limits, in addition to the
            mean
        nsamples : int, default=12
            Number of CDF samples to improve `mean` and `std` estimate.

        Returns
        -------
        limit(s) : float, array, or tuple of floats or arrays
            Returns the credibility interval boundaries for each column.
            By default, returns the mean over ``nsamples`` samples, which is
            either two numbers (``method='iso-pdf'``/``'equal-tailed'``) or
            one number (``method='lower-limit'``/``'upper-limit'``). If
            ``return_covariance=True``, returns a tuple (means, covariances)
            where covariances are the covariance over the sampled limits for
            each column.
        """
        if 'lower' in method:
            limits = ['lower']
        elif 'upper' in method:
            limits = ['upper']
        else:
            limits = ['lower', 'upper']
        cis = [credibility_interval(self[col], weights=self.get_weights(),
                                    level=level, method=method,
                                    return_covariance=return_covariance,
                                    nsamples=nsamples) for col in self.columns]
        if return_covariance:
            cis, covs = zip(*cis)
            mulidx = MultiIndex.from_product([
                self.columns.get_level_values(level=0),
                limits
            ])
            ncol = len(self.columns)
            nlim = len(limits)
            covs = np.asarray(covs).reshape(nlim*ncol, nlim).T
            covs = DataFrame(covs, index=limits, columns=mulidx)
        cis = np.atleast_2d(cis) if 'limit' in method else np.asarray(cis).T
        cis = DataFrame(data=cis, index=limits, columns=self.columns)
        if return_covariance:
            return cis, covs
        else:
            return cis

    @property
    def _constructor_sliced(self):
        return WeightedSeries

    @property
    def _constructor(self):
        return WeightedDataFrame

    def groupby(
        self,
        by=None,
        axis=no_default,
        level=None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True,
    ):  # pragma: no cover  # noqa: D102
        if axis is not lib.no_default:
            axis = self._get_axis_number(axis)
            if axis == 1:
                warnings.warn(
                    "DataFrame.groupby with axis=1 is deprecated. Do "
                    "`frame.T.groupby(...)` without axis instead.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            else:
                warnings.warn(
                    "The 'axis' keyword in DataFrame.groupby is deprecated "
                    "and will be removed in a future version.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
        else:
            axis = 0

        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")

        return WeightedDataFrameGroupBy(
            obj=self,
            keys=by,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )


for cls in [WeightedDataFrame, WeightedSeries, WeightedGroupBy,
            WeightedDataFrameGroupBy, WeightedSeriesGroupBy]:
    adjust_docstrings(cls, r'\bDataFrame\b', 'WeightedDataFrame')
    adjust_docstrings(cls, r'\bDataFrames\b', 'WeightedDataFrames')
    adjust_docstrings(cls, r'\bSeries\b', 'WeightedSeries')
    adjust_docstrings(cls, r'\bDataFrameGroupBy\b', 'WeightedDataFrameGroupBy')
    adjust_docstrings(cls, r'\bSeriesGroupBy\b', 'WeightedSeriesGroupBy')
    adjust_docstrings(cls, 'core.window.ewm', 'pandas.api.typing')
    adjust_docstrings(cls, 'core.window.expanding', 'pandas.api.typing')
    adjust_docstrings(cls, 'core.window.rolling', 'pandas.api.typing')
    adjust_docstrings(cls, 'core.window', 'pandas.api.typing')
adjust_docstrings(WeightedDataFrame, 'resample', 'pandas.DataFrame.resample')
adjust_docstrings(WeightedSeries,    'resample', 'pandas.Series.resample')
