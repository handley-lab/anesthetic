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

    mean = df.mean(axis=1)
    assert isinstance(mean, WeightedSeries)
    assert_allclose(mean.mean(), 0.5, atol=1e-2)


def test_WeightedDataFrame_std():
    df = test_WeightedDataFrame_constructor()
    std = df.std()
    assert isinstance(std, Series)
    assert_allclose(std, (1./12)**0.5, atol=1e-2)

    std = df.std(axis=1)
    assert isinstance(std, WeightedSeries)
    assert_allclose(std.mean(), (1./12)**0.5, atol=1e-1)


def test_WeightedDataFrame_cov():
    df = test_WeightedDataFrame_constructor()
    cov = df.cov()
    assert isinstance(cov, DataFrame)
    assert_allclose(cov, (1./12)*np.identity(3), atol=1e-2)


def test_WeightedDataFrame_corr():
    df = test_WeightedDataFrame_constructor()
    corr = df.corr()
    assert isinstance(corr, DataFrame)
    assert_allclose(corr, np.identity(3), atol=1e-2)


def test_WeightedDataFrame_corrwith():
    df = test_WeightedDataFrame_constructor()

    correl = df.corrwith(df.A)
    assert isinstance(correl, Series)
    assert_allclose(df.corrwith(df.A), df.corr()['A'])

    correl = df.corrwith(df[['A', 'B']])
    assert isinstance(correl, Series)
    assert_allclose(correl['A'], 1, atol=1e-2)
    assert_allclose(correl['B'], 1, atol=1e-2)
    assert np.isnan(correl['C'])


def test_WeightedDataFrame_median():
    df = test_WeightedDataFrame_constructor()
    median = df.median()
    assert isinstance(median, Series)
    assert_allclose(median, 0.5, atol=1e-2)

    median = df.median(axis=1)
    assert isinstance(median, WeightedSeries)
    assert_allclose(median.mean(), 0.5, atol=1e-2)


def test_WeightedDataFrame_sem():
    df = test_WeightedDataFrame_constructor()
    sem = df.sem()
    assert isinstance(sem, Series)
    assert_allclose(sem, (1./12)**0.5/np.sqrt(df.neff()), atol=1e-2)

    sem = df.sem(axis=1)
    assert isinstance(sem, WeightedSeries)


def test_WeightedDataFrame_kurtosis():
    df = test_WeightedDataFrame_constructor()
    kurtosis = df.kurtosis()
    assert isinstance(kurtosis, Series)
    assert_allclose(kurtosis, 9./5, atol=1e-2)
    assert_array_equal(df.kurtosis(), df.kurt())

    kurtosis = df.kurtosis(axis=1)
    assert isinstance(kurtosis, WeightedSeries)
    assert_array_equal(df.kurtosis(axis=1), df.kurt(axis=1))


def test_WeightedDataFrame_skew():
    df = test_WeightedDataFrame_constructor()
    skew = df.skew()
    assert isinstance(skew, Series)
    assert_allclose(skew, 0., atol=2e-2)

    skew = df.skew(axis=1)
    assert isinstance(skew, Series)


def test_WeightedDataFrame_mad():
    df = test_WeightedDataFrame_constructor()
    mad = df.mad()
    assert isinstance(mad, Series)
    assert_allclose(mad, 0.25, atol=1e-2)

    mad = df.mad(axis=1)
    assert isinstance(mad, Series)


def test_WeightedDataFrame_quantile():
    df = test_WeightedDataFrame_constructor()

    quantile = df.quantile()
    assert isinstance(quantile, Series)
    assert_allclose(quantile, 0.5, atol=1e-2)

    quantile = df.quantile(axis=1)
    assert isinstance(quantile, WeightedSeries)
    assert_allclose(quantile.mean(), 0.5, atol=1e-2)

    qs = np.linspace(0, 1, 10)
    for q in qs:
        quantile = df.quantile(q)
        assert isinstance(quantile, Series)
        assert_allclose(quantile, q, atol=1e-2)

        quantile = df.quantile(q, axis=1)
        assert isinstance(quantile, WeightedSeries)

    assert_allclose(df.quantile(qs), np.transpose([qs, qs, qs]), atol=1e-2)
    quantile = df.quantile(qs, axis=1)
    assert isinstance(quantile, WeightedDataFrame)

    with pytest.raises(NotImplementedError):
        df.quantile(numeric_only=False)


def test_WeightedDataFrame_hist():
    plt.figure()
    df = test_WeightedDataFrame_constructor()
    axes = df[['A', 'B']].hist(bins=20, density=True)
    for ax in axes.flatten():
        assert len(ax.patches) == 20
        norm = 0
        for patch in ax.patches:
            norm += patch.get_height() * patch.get_width()
        assert norm == pytest.approx(1)
    plt.close("all")


def test_WeightedDataFrame_sample():
    df = test_WeightedDataFrame_constructor()
    sample = df.sample()
    assert isinstance(sample, WeightedDataFrame)
    samples = df.sample(5)
    assert isinstance(samples, WeightedDataFrame)
    assert len(samples) == 5


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
    assert ~df.mean(axis=1).isna().any()
    assert_array_equal(df.mean(skipna=False).isna(), [True, False, False])
    assert_array_equal(df.mean(axis=1, skipna=False).isna()[0:6],
                       [True, False, False, False, False, False])

    assert ~df.std().isna().any()
    assert ~df.std(axis=1).isna().any()
    assert_array_equal(df.std(skipna=False).isna(), [True, False, False])
    assert_array_equal(df.std(axis=1, skipna=False).isna()[0:6],
                       [True, False, False, False, False, False])

    assert ~df.cov().isna().any().any()
    assert_array_equal(df.cov(skipna=False).isna(), [[True, True, True],
                                                     [True, False, False],
                                                     [True, False, False]])

    df['B'][2] = np.nan
    assert ~df.mean().isna().any()
    assert_array_equal(df.mean(skipna=False).isna(), [True, True, False])
    assert_array_equal(df.mean(axis=1, skipna=False).isna()[0:6],
                       [True, False, True, False, False, False])

    assert ~df.std().isna().any()
    assert_array_equal(df.std(skipna=False).isna(), [True, True, False])
    assert_array_equal(df.std(axis=1, skipna=False).isna()[0:6],
                       [True, False, True, False, False, False])

    assert ~df.cov().isna().any().any()
    assert_array_equal(df.cov(skipna=False).isna(), [[True, True, True],
                                                     [True, True, True],
                                                     [True, True, False]])

    df['C'][4] = np.nan
    assert ~df.mean().isna().any()
    assert df.mean(skipna=False).isna().all()
    assert_array_equal(df.mean(axis=1, skipna=False).isna()[0:6],
                       [True, False, True, False, True, False])

    assert ~df.std().isna().any()
    assert df.std(skipna=False).isna().all()
    assert_array_equal(df.std(axis=1, skipna=False).isna()[0:6],
                       [True, False, True, False, True, False])

    assert ~df.cov().isna().any().any()
    assert df.cov(skipna=False).isna().all().all()

    assert_allclose(df.mean(), 0.5, atol=1e-2)
    assert_allclose(df.std(), (1./12)**0.5, atol=1e-2)
    assert_allclose(df.cov(), (1./12)*np.identity(3), atol=1e-2)


def test_WeightedSeries_mean():
    series = test_WeightedSeries_constructor()
    series[0] = np.nan
    series.var(skipna=False)
    mean = series.mean()
    assert isinstance(mean, float)
    assert_allclose(mean, 0.5, atol=1e-2)


def test_WeightedSeries_std():
    series = test_WeightedSeries_constructor()
    std = series.std()
    assert isinstance(std, float)
    assert_allclose(std, (1./12)**0.5, atol=1e-2)

    series[0] = np.nan
    assert ~np.isnan(series.std())
    assert np.isnan(series.std(skipna=False))


def test_WeightedSeries_cov():
    df = test_WeightedDataFrame_constructor()
    assert_allclose(df.A.cov(df.A), 1./12, atol=1e-2)
    assert_allclose(df.A.cov(df.B), 0, atol=1e-2)

    df.loc[0, 'B'] = np.nan
    assert ~np.isnan(df.A.cov(df.B))
    assert np.isnan(df.A.cov(df.B, skipna=False))
    assert ~np.isnan(df.B.cov(df.A))
    assert np.isnan(df.B.cov(df.A, skipna=False))


def test_WeightedSeries_corr():
    df = test_WeightedDataFrame_constructor()
    assert_allclose(df.A.corr(df.A), 1., atol=1e-2)
    assert_allclose(df.A.corr(df.B), 0, atol=1e-2)
    D = df.A + df.B
    assert_allclose(df.A.corr(D), 1/np.sqrt(2), atol=1e-2)


def test_WeightedSeries_median():
    series = test_WeightedSeries_constructor()
    median = series.median()
    assert isinstance(median, float)
    assert_allclose(median, 0.5, atol=1e-2)


def test_WeightedSeries_sem():
    series = test_WeightedSeries_constructor()
    sem = series.sem()
    assert isinstance(sem, float)
    assert_allclose(sem, (1./12)**0.5/np.sqrt(series.neff()), atol=1e-2)


def test_WeightedSeries_kurtosis():
    series = test_WeightedSeries_constructor()
    kurtosis = series.kurtosis()
    assert isinstance(kurtosis, float)
    assert_allclose(kurtosis, 9./5, atol=1e-2)
    assert series.kurtosis() == series.kurt()

    series[0] = np.nan
    assert ~np.isnan(series.kurtosis())
    assert np.isnan(series.kurtosis(skipna=False))


def test_WeightedSeries_skew():
    series = test_WeightedSeries_constructor()
    skew = series.skew()
    assert isinstance(skew, float)
    assert_allclose(skew, 0., atol=1e-2)

    series[0] = np.nan
    assert ~np.isnan(series.skew())
    assert np.isnan(series.skew(skipna=False))


def test_WeightedSeries_mad():
    series = test_WeightedSeries_constructor()
    mad = series.mad()
    assert isinstance(mad, float)
    assert_allclose(mad, 0.25, atol=1e-2)

    series[0] = np.nan
    assert ~np.isnan(series.mad())
    assert np.isnan(series.mad(skipna=False))


def test_WeightedSeries_quantile():
    series = test_WeightedSeries_constructor()

    quantile = series.quantile()
    assert isinstance(quantile, float)
    assert_allclose(quantile, 0.5, atol=1e-2)

    qs = np.linspace(0, 1, 10)
    for q in qs:
        quantile = series.quantile(q)
        assert isinstance(quantile, float)
        assert_allclose(quantile, q, atol=1e-2)

    assert_allclose(series.quantile(qs), qs, atol=1e-2)

    with pytest.raises(NotImplementedError):
        series.quantile(numeric_only=False)


def test_WeightedSeries_hist():
    plt.figure()
    series = test_WeightedSeries_constructor()
    ax = series.hist(bins=20, density=True)
    assert len(ax.patches) == 20
    norm = 0
    for patch in ax.patches:
        norm += patch.get_height() * patch.get_width()
    assert norm == pytest.approx(1)
    plt.close("all")


def test_WeightedSeries_sample():
    series = test_WeightedSeries_constructor()
    sample = series.sample()
    assert isinstance(sample, WeightedSeries)
    samples = series.sample(5)
    assert isinstance(samples, WeightedSeries)
    assert len(samples) == 5


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
