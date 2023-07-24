# flake8: noqa
from pandas.core.indexes.multi import sparsify_labels
from pandas.io.formats.format import (
    DataFrameFormatter as DataFrameFormatter,
    _make_fixed_width, is_numeric_dtype
)
from pandas import MultiIndex


class _DataFrameFormatter(DataFrameFormatter):

    def _get_formatted_column_labels(self, frame):
        columns = frame.columns
        if isinstance(columns, MultiIndex):
            fmt_columns = columns.format(sparsify=False, adjoin=False)
            fmt_columns = list(zip(*fmt_columns))
            dtypes = self.frame.dtypes._values

            # if we have a Float level, they don't use leading space at all
            restrict_formatting = any(level.is_floating for level in columns.levels)
            need_leadsp = dict(zip(fmt_columns, map(is_numeric_dtype, dtypes)))

            def space_format(x, y):
                if (
                    y not in self.formatters
                    and need_leadsp[x]
                    and not restrict_formatting
                ): return " " + y
                return y

            str_columns = list(
                zip(*([space_format(x, y) for y in x] for x in fmt_columns))
            )
            if self.sparsify and len(str_columns):
                str_columns = sparsify_labels(str_columns)

            str_columns = [list(x) for x in zip(*str_columns)]
            str_columns = [_make_fixed_width(x) for x in str_columns]
        else:
            fmt_columns = columns.format()
            dtypes = self.frame.dtypes
            need_leadsp = dict(zip(fmt_columns, map(is_numeric_dtype, dtypes)))
            str_columns = [
                [" " + x if not self._get_formatter(i) and need_leadsp[x] else x]
                for i, x in enumerate(fmt_columns)
            ]
            str_columns = [_make_fixed_width(x) for x in str_columns]
        # self.str_columns = str_columns
        return str_columns
