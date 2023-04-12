from wpandas.core.base import _WeightedObject
from wpandas.core.util._code_transforms import adjust_weighted_docstrings

import warnings
from inspect import signature
import numpy as np
from pandas import Series, DataFrame, concat
from pandas._libs import lib
from pandas._libs.lib import no_default
from pandas.util._exceptions import find_stack_level
from numpy.ma import masked_array
from wpandas.core.util.weights import compress_weights


class WeightedDataFrame(_WeightedObject, DataFrame):
    """Weighted version of :class:`pandas.DataFrame`."""

    def mean(self, axis=0, skipna=True, *args, **kwargs):  # noqa: D102
        if self.isweighted(axis):
            if self.get_weights(axis).sum() == 0:
                return self._constructor_sliced(np.nan,
                                                index=self._get_axis(1-axis))
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
            if self.get_weights(axis).sum() == 0:
                return self._constructor_sliced(np.nan,
                                                index=self._get_axis(1-axis))
            null = self.isnull() & skipna
            mean = self.mean(axis=axis, skipna=skipna)
            var = np.average(masked_array((self-mean)**2, null),
                             weights=self.get_weights(axis), axis=axis)
            return self._constructor_sliced(var, index=self._get_axis(1-axis))
        else:
            return super().var(axis=axis, skipna=skipna, *args, **kwargs)

    def cov(self, *args, **kwargs):  # noqa: D102
        if self.isweighted():
            null = self.isnull()
            mean = self.mean(skipna=True)
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
        axis = self._get_axis_number(axis)
        if not self.isweighted(axis):
            return super().corrwith(other, drop=drop, axis=axis, method=method,
                                    *args, **kwargs)
        else:
            if isinstance(other, Series):
                answer = self.apply(lambda x: other.corr(x, method=method),
                                    axis=axis)
                return self._constructor_sliced(answer)

            left, right = self.align(other, join="inner", copy=False)

            if axis == 1:
                left = left.T
                right = right.T

            weights = left.index.to_frame()['weights']
            weights, _ = weights.align(right, join="inner", copy=False)

            # mask missing values
            left = left + right * 0
            right = right + left * 0

            # demeaned data
            ldem = left - left.mean()
            rdem = right - right.mean()

            num = (ldem * rdem * weights.to_numpy()[:, None]).sum()
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

    def kurt(self, axis=0, skipna=True, *args, **kwargs):  # noqa: D102
        if self.isweighted(axis):
            if self.get_weights(axis).sum() == 0:
                return self._constructor_sliced(np.nan,
                                                index=self._get_axis(1-axis))
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
            if self.get_weights(axis).sum() == 0:
                return self._constructor_sliced(np.nan,
                                                index=self._get_axis(1-axis))
            null = self.isnull() & skipna
            mean = self.mean(axis=axis, skipna=skipna)
            std = self.std(axis=axis, skipna=skipna)
            skew = np.average(masked_array(((self-mean)/std)**3, null),
                              weights=self.get_weights(axis), axis=axis)
            return self._constructor_sliced(skew, index=self._get_axis(1-axis))
        else:
            return super().skew(axis=axis, skipna=skipna, *args, **kwargs)

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
        from wpandas.core.series import WeightedSeries
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
        from wpandas.core.groupby import WeightedDataFrameGroupBy
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
            axis=axis,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )


adjust_weighted_docstrings(WeightedDataFrame)
