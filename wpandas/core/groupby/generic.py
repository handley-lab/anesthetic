"""
Define the WeightedSeriesGroupBy and WeightedDataFrameGroupBy
classes that hold the groupby interfaces (and some
implementations).

These are user facing as the result of the
``wdf.groupby(...)`` operations, which here returns a
WeightedDataFrameGroupBy object.
"""
from pandas.core.groupby import SeriesGroupBy, DataFrameGroupBy
from wpandas.core.groupby.groupby import WeightedGroupBy
from wpandas.core.util._code_transforms import adjust_weighted_docstrings


class WeightedSeriesGroupBy(WeightedGroupBy, SeriesGroupBy):
    """Weighted version of ``pandas.core.groupby.SeriesGroupBy``."""

    def sample(self, *args, **kwargs):  # noqa: D102
        return super().sample(weights=self.obj.get_weights(), *args, **kwargs)


class WeightedDataFrameGroupBy(WeightedGroupBy, DataFrameGroupBy):
    """Weighted version of ``pandas.core.groupby.DataFrameGroupBy``."""

    def get_weights(self):
        """Return the weights of the grouped samples."""
        return super().get_weights().min(axis=1-self.axis)

    def _gotitem(self, key, ndim: int, subset=None):  # pragma: no cover
        if ndim == 2:
            if subset is None:
                subset = self.obj
            return WeightedDataFrameGroupBy(
                subset,
                self.grouper,
                axis=self.axis,
                level=self.level,
                grouper=self.grouper,
                exclusions=self.exclusions,
                selection=key,
                as_index=self.as_index,
                sort=self.sort,
                group_keys=self.group_keys,
                squeeze=self.squeeze,
                observed=self.observed,
                mutated=self.mutated,
                dropna=self.dropna,
            )
        elif ndim == 1:
            if subset is None:
                subset = self.obj[key]
            return WeightedSeriesGroupBy(
                subset,
                level=self.level,
                grouper=self.grouper,
                selection=key,
                sort=self.sort,
                group_keys=self.group_keys,
                squeeze=self.squeeze,
                observed=self.observed,
                dropna=self.dropna,
            )

        raise AssertionError("invalid ndim for _gotitem")

    def sample(self, *args, **kwargs):  # noqa: D102
        return super().sample(weights=self.obj.get_weights(), *args, **kwargs)


adjust_weighted_docstrings(WeightedSeriesGroupBy)
