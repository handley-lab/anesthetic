import pytest
from anesthetic.weighted_pandas import WeightedDataFrame, WeightedSeries
from pandas import Series, DataFrame
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


def test_WeightedSeries_constructor():
    np.random.seed(0)
    N = 100000
    data = np.random.rand(N)

    series = WeightedSeries(data)
    assert_array_equal(series.weight, 1)
    assert_array_equal(series, data)

    with pytest.warns(FutureWarning):
        series = WeightedSeries(data, w=None)
        assert_array_equal(series.weight, 1)
        assert_array_equal(series, data)

    series = WeightedSeries(data, weight=None)
    assert_array_equal(series.weight, 1)
    assert_array_equal(series, data)

    weight = np.random.rand(N)
    with pytest.warns(FutureWarning):
        series = WeightedSeries(data, w=weight)
        assert_array_equal(series, data)

    series = WeightedSeries(data, weight=weight)
    assert_array_equal(series, data)

    assert series.weight.shape == (N,)
    assert series.shape == (N,)
    assert isinstance(series.weight, Series)
    assert_array_equal(series, data)
    assert_array_equal(series.weight, weight)
    assert isinstance(series.to_frame(), WeightedDataFrame)
    assert_array_equal(series.to_frame().weight, weight)

    return series


def test_WeightedDataFrame_constructor():
    np.random.seed(0)
    N = 100000
    m = 3
    data = np.random.rand(N, m)
    cols = ['A', 'B', 'C']

    df = WeightedDataFrame(data, columns=cols)
    assert_array_equal(df.weight, 1)
    assert_array_equal(df, data)

    with pytest.warns(FutureWarning):
        df = WeightedDataFrame(data, w=None, columns=cols)
        assert_array_equal(df.weight, 1)
        assert_array_equal(df, data)

    df = WeightedDataFrame(data, weight=None, columns=cols)
    assert_array_equal(df.weight, 1)
    assert_array_equal(df, data)

    weight = np.random.rand(N)
    with pytest.warns(FutureWarning):
        df = WeightedDataFrame(data, w=weight, columns=cols)
        assert df.weight.shape == (N,)
        assert df.shape == (N, m)
        assert isinstance(df.weight, Series)
        assert_array_equal(df, data)
        assert_array_equal(df.weight, weight)
        assert_array_equal(df.columns, cols)

    df = WeightedDataFrame(data, weight=weight, columns=cols)
    assert df.weight.shape == (N,)
    assert df.shape == (N, m)
    assert isinstance(df.weight, Series)
    assert_array_equal(df, data)
    assert_array_equal(df.weight, weight)
    assert_array_equal(df.columns, cols)
    return df

    return df


def test_WeightedDataFrame_slice():
    df = test_WeightedDataFrame_constructor()
    assert isinstance(df['A'], WeightedSeries)
    assert df[:10].shape == (10, 3)
    assert df[:10].weight.shape == (10,)
    assert df[:10]._rand.shape == (10,)


def test_WeightedDataFrame_mean():
    df = test_WeightedDataFrame_constructor()
    mean = df.mean()
    assert isinstance(mean, Series)
    assert_allclose(mean, 0.5, atol=1e-2)


def test_WeightedDataFrame_std():
    df = test_WeightedDataFrame_constructor()
    std = df.std()
    assert isinstance(std, Series)
    assert_allclose(std, (1./12)**0.5, atol=1e-2)


def test_WeightedDataFrame_cov():
    df = test_WeightedDataFrame_constructor()
    cov = df.cov()
    assert isinstance(cov, DataFrame)
    assert_allclose(cov, (1./12)*np.identity(3), atol=1e-2)


def test_WeightedDataFrame_median():
    df = test_WeightedDataFrame_constructor()
    median = df.median()
    assert isinstance(median, Series)
    assert_allclose(median, 0.5, atol=1e-2)


def test_WeightedDataFrame_quantile():
    df = test_WeightedDataFrame_constructor()
    for q in np.linspace(0, 1, 10):
        quantile = df.quantile(q)
        assert isinstance(quantile, Series)
        assert_allclose(quantile, q, atol=1e-2)


def test_WeightedDataFrame_neff():
    df = test_WeightedDataFrame_constructor()
    neff = df.neff()
    assert isinstance(neff, float)
    assert neff < len(df)
    assert neff > len(df) * np.exp(-0.25)


def test_WeightedDataFrame_compress():
    df = test_WeightedDataFrame_constructor()
    assert_allclose(df.neff(), len(df.compress()), rtol=1e-2)
    for i in np.logspace(3, 5, 10):
        assert_allclose(i, len(df.compress(i)), rtol=1e-1)
    unit_weights = df.compress(0)
    assert(len(np.unique(unit_weights.index)) == len(unit_weights))


def test_WeightedSeries_mean():
    series = test_WeightedSeries_constructor()
    mean = series.mean()
    assert isinstance(mean, float)
    assert_allclose(mean, 0.5, atol=1e-2)


def test_WeightedSeries_std():
    series = test_WeightedSeries_constructor()
    std = series.std()
    assert isinstance(std, float)
    assert_allclose(std, (1./12)**0.5, atol=1e-2)


def test_WeightedSeries_median():
    series = test_WeightedSeries_constructor()
    median = series.median()
    assert isinstance(median, float)
    assert_allclose(median, 0.5, atol=1e-2)


def test_WeightedSeries_quantile():
    series = test_WeightedSeries_constructor()
    for q in np.linspace(0, 1, 10):
        quantile = series.quantile(q)
        assert isinstance(quantile, float)
        assert_allclose(quantile, q, atol=1e-2)


def test_WeightedSeries_neff():
    series = test_WeightedSeries_constructor()
    neff = series.neff()
    assert isinstance(neff, float)
    assert neff < len(series)
    assert neff > len(series) * np.exp(-0.25)


def test_WeightedSeries_compress():
    series = test_WeightedSeries_constructor()
    assert_allclose(series.neff(), len(series.compress()), rtol=1e-2)
    for i in np.logspace(3, 5, 10):
        assert_allclose(i, len(series.compress(i)), rtol=1e-1)
    unit_weights = series.compress(0)
    assert(len(np.unique(unit_weights.index)) == len(unit_weights))
