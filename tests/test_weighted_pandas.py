from anesthetic.weighted_pandas import WeightedDataFrame, WeightedSeries
from pandas import Series
import numpy
from numpy.testing import assert_array_equal


def test_WeightedDataFrame_constructor():
    numpy.random.seed(0)
    N = 1000
    m = 3
    data = numpy.random.rand(N, m)
    weights = numpy.random.rand(N)
    cols = ['A', 'B', 'C']
    df = WeightedDataFrame(data, w=weights, columns=cols)
    assert df.w.shape == (N,)
    assert df.shape == (N, m)
    assert isinstance(df.w, Series)
    assert_array_equal(df, data)
    assert_array_equal(df.w, weights)
    assert_array_equal(df.columns, cols)
    return df


def test_WeightedDataFrame_slice():
    df = test_WeightedDataFrame_constructor()
    assert isinstance(df['A'],  WeightedSeries)
    assert df[:10].shape == (10, 3)
    assert df[:10].w.shape == (10,)
    assert df[:10].u.shape == (10,)
