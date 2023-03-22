import pandas
import lpandas.core
from lpandas._format import _DataFrameFormatter

pandas.io.formats.format.DataFrameFormatter = _DataFrameFormatter
pandas.options.display.max_colwidth = 14

LabelledSeries = lpandas.core.LabelledSeries
LabelledDataFrame = lpandas.core.LabelledDataFrame
