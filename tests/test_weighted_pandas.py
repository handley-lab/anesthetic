from anesthetic.weighted_pandas import WeightedDataFrame, WeightedSeries
from pandas import Series, DataFrame
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import matplotlib.pyplot as plt


def test_WeightedSeries_constructor():
    np.random.seed(0)
    N = 100000
    data = np.random.rand(N)

    series = WeightedSeries(data)
    assert_array_equal(series.weights, 1)
    assert_array_equal(series, data)

    series = WeightedSeries(data, weights=None)
    assert_array_equal(series.weights, 1)
    assert_array_equal(series, data)

    weights = np.random.rand(N)
    series = WeightedSeries(data, weights=weights)
    assert_array_equal(series, data)

    assert series.weights.shape == (N,)
    assert series.shape == (N,)
    assert isinstance(series.weights, np.ndarray)
    assert_array_equal(series, data)
    assert_array_equal(series.weights, weights)
    assert isinstance(series.to_frame(), WeightedDataFrame)
    assert_array_equal(series.to_frame().weights, weights)

    return series


def test_WeightedDataFrame_constructor():
    np.random.seed(0)
    N = 100000
    m = 3
    data = np.random.rand(N, m)
    cols = ['A', 'B', 'C']

    df = WeightedDataFrame(data, columns=cols)
    assert_array_equal(df.weights, 1)
    assert_array_equal(df, data)

    df = WeightedDataFrame(data, weights=None, columns=cols)
    assert_array_equal(df.weights, 1)
    assert_array_equal(df, data)

    weights = np.random.rand(N)
    df = WeightedDataFrame(data, weights=weights, columns=cols)
    assert df.weights.shape == (N,)
    assert df.shape == (N, m)
    assert isinstance(df.weights, np.ndarray)
    assert_array_equal(df, data)
    assert_array_equal(df.weights, weights)
    assert_array_equal(df.columns, cols)
    return df


def test_WeightedDataFrame_key():
    df = test_WeightedDataFrame_constructor()
    for key1 in df.columns:
        assert_array_equal(df.weights, df[key1].weights)
        for key2 in df.columns:
            assert_array_equal(df[key1].weights, df[key2].weights)


def test_WeightedDataFrame_slice():
    df = test_WeightedDataFrame_constructor()
    assert isinstance(df['A'], WeightedSeries)
    assert df[:10].shape == (10, 3)
    assert df[:10].weights.shape == (10,)
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


def test_WeightedDataFrame_hist():
    df = test_WeightedDataFrame_constructor()
    axes = df[['A', 'B']].hist(bins=20, density=True)
    for ax in axes.flatten():
        assert len(ax.patches) == 20
        norm = 0
        for patch in ax.patches:
            norm += patch.get_height() * patch.get_width()
        assert norm == pytest.approx(1)
    plt.close("all")


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
    assert_array_equal(df.compress(), df.compress())
    assert_array_equal(df.compress(i), df.compress(i))
    assert_array_equal(df.compress(-1), df.compress(-1))


def test_WeightedDataFrame_nan():
    df = test_WeightedDataFrame_constructor()

    df['A'][0] = np.nan
    assert ~df.mean().isna().any()
    assert_array_equal(df.mean(skipna=False).isna(), [True, False, False])

    assert ~df.std().isna().any()
    assert_array_equal(df.std(skipna=False).isna(), [True, False, False])

    assert ~df.cov().isna().any().any()
    assert_array_equal(df.cov(skipna=False).isna(), [[True, True, True],
                                                     [True, False, False],
                                                     [True, False, False]])

    df['B'][1] = np.nan
    assert ~df.mean().isna().any()
    assert_array_equal(df.mean(skipna=False).isna(), [True, True, False])

    assert ~df.std().isna().any()
    assert_array_equal(df.std(skipna=False).isna(), [True, True, False])

    assert ~df.cov().isna().any().any()
    assert_array_equal(df.cov(skipna=False).isna(), [[True, True, True],
                                                     [True, True, True],
                                                     [True, True, False]])

    df['C'][2] = np.nan
    assert ~df.mean().isna().any()
    assert df.mean(skipna=False).isna().all()

    assert ~df.std().isna().any()
    assert df.std(skipna=False).isna().all()

    assert ~df.cov().isna().any().any()
    assert df.cov(skipna=False).isna().all().all()

    assert_allclose(df.mean(), 0.5, atol=1e-2)
    assert_allclose(df.std(), (1./12)**0.5, atol=1e-2)
    assert_allclose(df.cov(), (1./12)*np.identity(3), atol=1e-2)


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


def test_WeightedSeries_hist():
    series = test_WeightedSeries_constructor()
    ax = series.hist(bins=20, density=True)
    assert len(ax.patches) == 20
    norm = 0
    for patch in ax.patches:
        norm += patch.get_height() * patch.get_width()
    assert norm == pytest.approx(1)
    plt.close("all")


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


def test_WeightedSeries_nan():
    series = test_WeightedSeries_constructor()

    series[0] = np.nan

    assert ~np.isnan(series.mean())
    assert np.isnan(series.mean(skipna=False))
    assert ~np.isnan(series.std())
    assert np.isnan(series.std(skipna=False))
    assert ~np.isnan(series.var())
    assert np.isnan(series.var(skipna=False))

    assert_allclose(series.mean(), 0.5, atol=1e-2)
    assert_allclose(series.var(), 1./12, atol=1e-2)
    assert_allclose(series.std(), (1./12)**0.5, atol=1e-2)
