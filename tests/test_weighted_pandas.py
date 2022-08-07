from anesthetic.weighted_pandas import WeightedDataFrame, WeightedSeries
from anesthetic.utils import channel_capacity
from pandas import Series, DataFrame
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix, bootstrap_plot
from pandas.plotting._matplotlib.misc import (
    scatter_matrix as orig_scatter_matrix
)


@pytest.fixture
def series():
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


@pytest.fixture
def frame():
    np.random.seed(0)
    N = 100000
    m = 3
    data = np.random.rand(N, m)
    cols = ['A', 'B', 'C']

    frame = WeightedDataFrame(data, columns=cols)
    assert_array_equal(frame.weights, 1)
    assert_array_equal(frame, data)

    frame = WeightedDataFrame(data, weights=None, columns=cols)
    assert_array_equal(frame.weights, 1)
    assert_array_equal(frame, data)

    weights = np.random.rand(N)
    frame = WeightedDataFrame(data, weights=weights, columns=cols)
    assert frame.weights.shape == (N,)
    assert frame.shape == (N, m)
    assert isinstance(frame.weights, np.ndarray)
    assert_array_equal(frame, data)
    assert_array_equal(frame.weights, weights)
    assert_array_equal(frame.columns, cols)
    return frame


def test_WeightedDataFrame_key(frame):
    for key1 in frame.columns:
        assert_array_equal(frame.weights, frame[key1].weights)
        for key2 in frame.columns:
            assert_array_equal(frame[key1].weights, frame[key2].weights)


def test_WeightedDataFrame_slice(frame):
    assert isinstance(frame['A'], WeightedSeries)
    assert frame[:10].shape == (10, 3)
    assert frame[:10].weights.shape == (10,)
    assert frame[:10]._rand.shape == (10,)


def test_WeightedDataFrame_mean(frame):
    mean = frame.mean()
    assert isinstance(mean, Series)
    assert_allclose(mean, 0.5, atol=1e-2)

    mean = frame.mean(axis=1)
    assert isinstance(mean, WeightedSeries)
    assert_allclose(mean.mean(), 0.5, atol=1e-2)


def test_WeightedDataFrame_std(frame):
    std = frame.std()
    assert isinstance(std, Series)
    assert_allclose(std, (1./12)**0.5, atol=1e-2)

    std = frame.std(axis=1)
    assert isinstance(std, WeightedSeries)
    assert_allclose(std.mean(), (1./12)**0.5, atol=1e-1)


def test_WeightedDataFrame_cov(frame):
    cov = frame.cov()
    assert isinstance(cov, DataFrame)
    assert_allclose(cov, (1./12)*np.identity(3), atol=1e-2)


def test_WeightedDataFrame_corr(frame):
    corr = frame.corr()
    assert isinstance(corr, DataFrame)
    assert_allclose(corr, np.identity(3), atol=1e-2)


def test_WeightedDataFrame_corrwith(frame):
    correl = frame.corrwith(frame.A)
    assert isinstance(correl, Series)
    assert_allclose(frame.corrwith(frame.A), frame.corr()['A'])

    correl = frame.corrwith(frame[['A', 'B']])
    assert isinstance(correl, Series)
    assert_allclose(correl['A'], 1, atol=1e-2)
    assert_allclose(correl['B'], 1, atol=1e-2)
    assert np.isnan(correl['C'])


def test_WeightedDataFrame_median(frame):
    median = frame.median()
    assert isinstance(median, Series)
    assert_allclose(median, 0.5, atol=1e-2)

    median = frame.median(axis=1)
    assert isinstance(median, WeightedSeries)
    assert_allclose(median.mean(), 0.5, atol=1e-2)


def test_WeightedDataFrame_sem(frame):
    sem = frame.sem()
    assert isinstance(sem, Series)
    assert_allclose(sem, (1./12)**0.5/np.sqrt(frame.neff()), atol=1e-2)

    sem = frame.sem(axis=1)
    assert isinstance(sem, WeightedSeries)


def test_WeightedDataFrame_kurtosis(frame):
    kurtosis = frame.kurtosis()
    assert isinstance(kurtosis, Series)
    assert_allclose(kurtosis, 9./5, atol=1e-2)
    assert_array_equal(frame.kurtosis(), frame.kurt())

    kurtosis = frame.kurtosis(axis=1)
    assert isinstance(kurtosis, WeightedSeries)
    assert_array_equal(frame.kurtosis(axis=1), frame.kurt(axis=1))


def test_WeightedDataFrame_skew(frame):
    skew = frame.skew()
    assert isinstance(skew, Series)
    assert_allclose(skew, 0., atol=2e-2)

    skew = frame.skew(axis=1)
    assert isinstance(skew, Series)


def test_WeightedDataFrame_mad(frame):
    mad = frame.mad()
    assert isinstance(mad, Series)
    assert_allclose(mad, 0.25, atol=1e-2)

    mad = frame.mad(axis=1)
    assert isinstance(mad, Series)


def test_WeightedDataFrame_quantile(frame):
    quantile = frame.quantile()
    assert isinstance(quantile, Series)
    assert_allclose(quantile, 0.5, atol=1e-2)

    quantile = frame.quantile(axis=1)
    assert isinstance(quantile, WeightedSeries)
    assert_allclose(quantile.mean(), 0.5, atol=1e-2)

    qs = np.linspace(0, 1, 10)
    for q in qs:
        quantile = frame.quantile(q)
        assert isinstance(quantile, Series)
        assert_allclose(quantile, q, atol=1e-2)

        quantile = frame.quantile(q, axis=1)
        assert isinstance(quantile, WeightedSeries)

    assert_allclose(frame.quantile(qs), np.transpose([qs, qs, qs]), atol=1e-2)
    quantile = frame.quantile(qs, axis=1)
    assert isinstance(quantile, WeightedDataFrame)

    with pytest.raises(NotImplementedError):
        frame.quantile(numeric_only=False)


def test_WeightedDataFrame_sample(frame):
    sample = frame.sample()
    assert isinstance(sample, WeightedDataFrame)
    samples = frame.sample(5)
    assert isinstance(samples, WeightedDataFrame)
    assert len(samples) == 5


def test_WeightedDataFrame_neff(frame):
    neff = frame.neff()
    assert isinstance(neff, float)
    assert neff < len(frame)
    assert neff > len(frame) * np.exp(-0.25)


def test_WeightedDataFrame_compress(frame):
    assert_allclose(frame.neff(), len(frame.compress()), rtol=1e-2)
    for i in np.logspace(3, 5, 10):
        assert_allclose(i, len(frame.compress(i)), rtol=1e-1)
    unit_weights = frame.compress(0)
    assert len(np.unique(unit_weights.index)) == len(unit_weights)
    assert_array_equal(frame.compress(), frame.compress())
    assert_array_equal(frame.compress(i), frame.compress(i))
    assert_array_equal(frame.compress(-1), frame.compress(-1))


def test_WeightedDataFrame_nan(frame):
    frame['A'][0] = np.nan
    assert ~frame.mean().isna().any()
    assert ~frame.mean(axis=1).isna().any()
    assert_array_equal(frame.mean(skipna=False).isna(), [True, False, False])
    assert_array_equal(frame.mean(axis=1, skipna=False).isna()[0:6],
                       [True, False, False, False, False, False])

    assert ~frame.std().isna().any()
    assert ~frame.std(axis=1).isna().any()
    assert_array_equal(frame.std(skipna=False).isna(), [True, False, False])
    assert_array_equal(frame.std(axis=1, skipna=False).isna()[0:6],
                       [True, False, False, False, False, False])

    assert ~frame.cov().isna().any().any()
    assert_array_equal(frame.cov(skipna=False).isna(), [[True, True, True],
                                                        [True, False, False],
                                                        [True, False, False]])

    frame['B'][2] = np.nan
    assert ~frame.mean().isna().any()
    assert_array_equal(frame.mean(skipna=False).isna(), [True, True, False])
    assert_array_equal(frame.mean(axis=1, skipna=False).isna()[0:6],
                       [True, False, True, False, False, False])

    assert ~frame.std().isna().any()
    assert_array_equal(frame.std(skipna=False).isna(), [True, True, False])
    assert_array_equal(frame.std(axis=1, skipna=False).isna()[0:6],
                       [True, False, True, False, False, False])

    assert ~frame.cov().isna().any().any()
    assert_array_equal(frame.cov(skipna=False).isna(), [[True, True, True],
                                                        [True, True, True],
                                                        [True, True, False]])

    frame['C'][4] = np.nan
    assert ~frame.mean().isna().any()
    assert frame.mean(skipna=False).isna().all()
    assert_array_equal(frame.mean(axis=1, skipna=False).isna()[0:6],
                       [True, False, True, False, True, False])

    assert ~frame.std().isna().any()
    assert frame.std(skipna=False).isna().all()
    assert_array_equal(frame.std(axis=1, skipna=False).isna()[0:6],
                       [True, False, True, False, True, False])

    assert ~frame.cov().isna().any().any()
    assert frame.cov(skipna=False).isna().all().all()

    assert_allclose(frame.mean(), 0.5, atol=1e-2)
    assert_allclose(frame.std(), (1./12)**0.5, atol=1e-2)
    assert_allclose(frame.cov(), (1./12)*np.identity(3), atol=1e-2)


def test_WeightedSeries_mean(series):
    series[0] = np.nan
    series.var(skipna=False)
    mean = series.mean()
    assert isinstance(mean, float)
    assert_allclose(mean, 0.5, atol=1e-2)


def test_WeightedSeries_std(series):
    std = series.std()
    assert isinstance(std, float)
    assert_allclose(std, (1./12)**0.5, atol=1e-2)

    series[0] = np.nan
    assert ~np.isnan(series.std())
    assert np.isnan(series.std(skipna=False))


def test_WeightedSeries_cov(frame):
    assert_allclose(frame.A.cov(frame.A), 1./12, atol=1e-2)
    assert_allclose(frame.A.cov(frame.B), 0, atol=1e-2)

    frame.loc[0, 'B'] = np.nan
    assert ~np.isnan(frame.A.cov(frame.B))
    assert np.isnan(frame.A.cov(frame.B, skipna=False))
    assert ~np.isnan(frame.B.cov(frame.A))
    assert np.isnan(frame.B.cov(frame.A, skipna=False))


def test_WeightedSeries_corr(frame):
    assert_allclose(frame.A.corr(frame.A), 1., atol=1e-2)
    assert_allclose(frame.A.corr(frame.B), 0, atol=1e-2)
    D = frame.A + frame.B
    assert_allclose(frame.A.corr(D), 1/np.sqrt(2), atol=1e-2)


def test_WeightedSeries_median(series):
    median = series.median()
    assert isinstance(median, float)
    assert_allclose(median, 0.5, atol=1e-2)


def test_WeightedSeries_sem(series):
    sem = series.sem()
    assert isinstance(sem, float)
    assert_allclose(sem, (1./12)**0.5/np.sqrt(series.neff()), atol=1e-2)


def test_WeightedSeries_kurtosis(series):
    kurtosis = series.kurtosis()
    assert isinstance(kurtosis, float)
    assert_allclose(kurtosis, 9./5, atol=1e-2)
    assert series.kurtosis() == series.kurt()

    series[0] = np.nan
    assert ~np.isnan(series.kurtosis())
    assert np.isnan(series.kurtosis(skipna=False))


def test_WeightedSeries_skew(series):
    skew = series.skew()
    assert isinstance(skew, float)
    assert_allclose(skew, 0., atol=1e-2)

    series[0] = np.nan
    assert ~np.isnan(series.skew())
    assert np.isnan(series.skew(skipna=False))


def test_WeightedSeries_mad(series):
    mad = series.mad()
    assert isinstance(mad, float)
    assert_allclose(mad, 0.25, atol=1e-2)

    series[0] = np.nan
    assert ~np.isnan(series.mad())
    assert np.isnan(series.mad(skipna=False))


def test_WeightedSeries_quantile(series):

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


def test_WeightedSeries_sample(series):
    sample = series.sample()
    assert isinstance(sample, WeightedSeries)
    samples = series.sample(5)
    assert isinstance(samples, WeightedSeries)
    assert len(samples) == 5


def test_WeightedSeries_neff(series):
    neff = series.neff()
    assert isinstance(neff, float)
    assert neff < len(series)
    assert neff > len(series) * np.exp(-0.25)


def test_WeightedSeries_compress(series):
    assert_allclose(series.neff(), len(series.compress()), rtol=1e-2)
    for i in np.logspace(3, 5, 10):
        assert_allclose(i, len(series.compress(i)), rtol=1e-1)
    unit_weights = series.compress(0)
    assert len(np.unique(unit_weights.index)) == len(unit_weights)


def test_WeightedSeries_nan(series):

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


@pytest.fixture
def mcmc_df():
    np.random.seed(0)
    m, s = 0.5, 0.1

    def logL(x):
        return x, -((x-m)**2).sum(axis=-1)/2/s**2

    np.random.seed(0)
    x0, logL0 = logL(np.random.normal(m, s, 4))
    dat = []
    for _ in range(3000):
        dat.append(x0)
        x1, logL1 = logL(np.random.normal(x0, s/2))
        if np.log(np.random.rand()) < logL1 - logL0:
            x0, logL0 = x1, logL1

    return DataFrame(dat, columns=["x", "y", "z", "w"])


@pytest.fixture
def mcmc_wdf(mcmc_df):
    weights = mcmc_df.groupby(mcmc_df.columns.tolist(), sort=False).size()
    return WeightedDataFrame(mcmc_df.drop_duplicates(), weights=weights.values)


def test_WeightedDataFrame_hist(mcmc_df, mcmc_wdf):
    df_axes = mcmc_df.hist()
    wdf_axes = mcmc_wdf.hist()
    for df_ax, wdf_ax in zip(df_axes.flatten(), wdf_axes.flatten()):
        for df_patch, wdf_patch in zip(df_ax.patches, wdf_ax.patches):
            assert df_patch.get_height() == wdf_patch.get_height()
            assert df_patch.get_width() == wdf_patch.get_width()
            assert df_patch.get_xy() == wdf_patch.get_xy()

    df_axes = mcmc_df.plot.hist(subplots=True)
    wdf_axes = mcmc_wdf.plot.hist(subplots=True)
    for df_ax, wdf_ax in zip(df_axes.flatten(), wdf_axes.flatten()):
        for df_patch, wdf_patch in zip(df_ax.patches, wdf_ax.patches):
            assert df_patch.get_height() == wdf_patch.get_height()
            assert df_patch.get_width() == wdf_patch.get_width()
            assert df_patch.get_xy() == wdf_patch.get_xy()

    plt.close("all")


def test_WeightedSeries_hist(mcmc_df, mcmc_wdf):

    fig, axes = plt.subplots(2)
    mcmc_df.x.hist(ax=axes[0])
    mcmc_wdf.x.hist(ax=axes[1])

    for df_patch, wdf_patch in zip(axes[0].patches, axes[1].patches):
        assert df_patch.get_height() == wdf_patch.get_height()
        assert df_patch.get_width() == wdf_patch.get_width()
        assert df_patch.get_xy() == wdf_patch.get_xy()

    fig, axes = plt.subplots(2)
    mcmc_df.x.plot.hist(ax=axes[0])
    mcmc_wdf.x.plot.hist(ax=axes[1])

    for df_patch, wdf_patch in zip(axes[0].patches, axes[1].patches):
        assert df_patch.get_height() == wdf_patch.get_height()
        assert df_patch.get_width() == wdf_patch.get_width()
        assert df_patch.get_xy() == wdf_patch.get_xy()

    plt.close("all")


def test_KdePlot(mcmc_df, mcmc_wdf):
    bw_method = 0.3
    fig, axes = plt.subplots(2)
    mcmc_df.x.plot.kde(bw_method=bw_method, ax=axes[0])
    mcmc_wdf.x.plot.kde(bw_method=bw_method, ax=axes[1])
    df_line, wdf_line = axes[0].lines[0], axes[1].lines[0]
    assert (df_line.get_xdata() == wdf_line.get_xdata()).all()
    assert_allclose(df_line.get_ydata(),  wdf_line.get_ydata(), atol=1e-4)

    plt.close("all")


def test_scatter_matrix(mcmc_df, mcmc_wdf):
    axes = scatter_matrix(mcmc_df)
    data = axes[0, 1].collections[0].get_offsets().data
    axes = orig_scatter_matrix(mcmc_df)
    orig_data = axes[0, 1].collections[0].get_offsets().data

    assert_allclose(data, orig_data)

    axes = scatter_matrix(mcmc_wdf)
    data = axes[0, 1].collections[0].get_offsets().data
    n = len(data)
    neff = channel_capacity(mcmc_wdf.weights)
    assert_allclose(n, neff, atol=np.sqrt(n))

    axes = orig_scatter_matrix(mcmc_wdf)
    orig_data = axes[0, 1].collections[0].get_offsets().data
    n = len(orig_data)
    assert n == len(mcmc_wdf)

    axes = scatter_matrix(mcmc_wdf, ncompress=50)
    n = len(axes[0, 1].collections[0].get_offsets().data)
    assert_allclose(n, 50, atol=np.sqrt(n))

    plt.close("all")


def test_bootstrap_plot(mcmc_df, mcmc_wdf):
    bootstrap_plot(mcmc_wdf.x)
    bootstrap_plot(mcmc_wdf.x, ncompress=500)
    plt.close("all")


def test_BoxPlot(mcmc_df, mcmc_wdf):
    mcmc_df.plot.box()
    mcmc_wdf.plot.box()

    mcmc_df.boxplot()
    mcmc_wdf.boxplot()

    plt.close("all")
    mcmc_df.x.plot.box()
    plt.close("all")
    mcmc_wdf.x.plot.box()

    mcmc_df.plot.box(subplots=True)
    mcmc_wdf.plot.box(subplots=True)

    mcmc_df['split'] = ''
    mcmc_df.loc[:len(mcmc_df)//2, 'split'] = 'A'
    mcmc_df.loc[len(mcmc_df)//2:, 'split'] = 'B'

    mcmc_wdf['split'] = ''
    mcmc_wdf.iloc[:len(mcmc_wdf)//2, -1] = 'A'
    mcmc_wdf.iloc[len(mcmc_wdf)//2:, -1] = 'B'

    mcmc_df.groupby('split').boxplot()
    mcmc_wdf.groupby('split').boxplot()

    plt.close("all")

    for return_type in ['dict', 'both']:
        mcmc_wdf.plot.box(return_type=return_type)
        mcmc_wdf.boxplot(return_type=return_type)

    mcmc_wdf.boxplot(xlabel='xlabel')
    mcmc_wdf.boxplot(ylabel='ylabel')

    mcmc_wdf.boxplot(vert=False)
    mcmc_wdf.boxplot(fontsize=30)


def test_ScatterPlot(mcmc_df, mcmc_wdf):
    mcmc_df.plot.scatter('x', 'y')
    ax = mcmc_wdf.plot.scatter('x', 'y')

    n = len(ax.collections[0].get_offsets().data)
    neff = channel_capacity(mcmc_wdf.weights)
    assert_allclose(n, neff, atol=np.sqrt(n))

    ax = mcmc_wdf.plot.scatter('x', 'y', ncompress=50)
    n = len(ax.collections[0].get_offsets().data)
    assert_allclose(n, 50, atol=np.sqrt(50))


def test_HexBinPLot(mcmc_df, mcmc_wdf):
    mcmc_df.plot.hexbin('x', 'y')
    mcmc_wdf.plot.hexbin('x', 'y')
