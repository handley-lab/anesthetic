"""
Provide the weighted groupby split-apply-combine paradigm. Define the
WeightedGroupBy class providing the base-class of operations.

The WeightedSeriesGroupBy and WeightedDataFrameGroupBy sub-class (defined in
wpandas.core.groupby.generic) expose these user-facing objects to provide
specific functionality.
"""

from pandas.core.groupby import GroupBy, ops
from wpandas.core.util._code_transforms import adjust_weighted_docstrings


class WeightedGroupBy(GroupBy):
    """Weighted version of ``pandas.core.groupby.GroupBy``."""

    grouper: ops.BaseGrouper
    """:meta private:"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _add_weights(self, name, *args, **kwargs):
        result = self.agg(lambda df: getattr(self.obj._constructor(df), name)
                          (*args, **kwargs)).set_weights(self.get_weights())
        return result.__finalize__(self.obj, method="groupby")

    def mean(self, *args, **kwargs):  # noqa: D102
        return self._add_weights("mean", *args, **kwargs)

    def std(self, *args, **kwargs):  # noqa: D102
        return self._add_weights("std", *args, **kwargs)

    def median(self, *args, **kwargs):  # noqa: D102
        return self._add_weights("median", *args, **kwargs)

    def var(self, *args, **kwargs):  # noqa: D102
        return self._add_weights("var", *args, **kwargs)

    def kurt(self, *args, **kwargs):  # noqa: D102
        return self._add_weights("kurt", *args, **kwargs)

    def kurtosis(self, *args, **kwargs):  # noqa: D102
        return self._add_weights("kurtosis", *args, **kwargs)

    def sem(self, *args, **kwargs):  # noqa: D102
        return self._add_weights("sem", *args, **kwargs)

    def quantile(self, *args, **kwargs):  # noqa: D102
        return self._add_weights("quantile", *args, **kwargs)

    def get_weights(self):
        """Return the weights of the grouped samples."""
        return self.agg(lambda df: df.get_weights().sum())

    def _make_wrapper(self, name):
        _wrapper = super()._make_wrapper(name)

        def wrapper(*args, **kwargs):
            result = _wrapper(*args, **kwargs)
            try:
                index = result.index.get_level_values(self.keys)
                weights = self.get_weights()[index]
            except KeyError:
                weights = self.get_weights()
            return result.set_weights(weights, level=1)

        wrapper.__name__ = name
        return wrapper


adjust_weighted_docstrings(WeightedGroupBy)
