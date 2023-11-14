from anesthetic.weighted_pandas import WeightedDataFrame, WeightedSeries
from pandas import DataFrame, MultiIndex
import pandas.testing
from anesthetic.utils import neff
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pandas.plotting import scatter_matrix, bootstrap_plot
from pandas.plotting._matplotlib.misc import (
    scatter_matrix as orig_scatter_matrix
)


@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    yield
    plt.close("all")


@pytest.fixture
def series():
    np.random.seed(0)
    N = 100000
    data = np.random.rand(N)

    series = WeightedSeries(data)
    assert_array_equal(series.get_weights(), 1)
    assert not series.isweighted()
    assert_array_equal(series, data)

    series = WeightedSeries(data, weights=None)
    assert_array_equal(series.get_weights(), 1)
    assert not series.isweighted()
    assert_array_equal(series, data)

    weights = np.random.rand(N)
    series = WeightedSeries(data, weights=weights)
    assert series.isweighted()
    assert_array_equal(series, data)

    assert series.get_weights().shape == (N,)
    assert series.shape == (N,)
    assert isinstance(series.get_weights(), np.ndarray)
    assert_array_equal(series, data)
    assert_array_equal(series.get_weights(), weights)
    assert isinstance(series.to_frame(), WeightedDataFrame)
    assert_array_equal(series.to_frame().get_weights(), weights)
    assert_array_equal(series.index.get_level_values('weights'), weights)

    return series


@pytest.fixture
def frame():
    np.random.seed(0)
    N = 100000
    cols = ['A', 'B', 'C', 'D', 'E', 'F']
    m = len(cols)
    data = np.random.rand(N, m)

    frame = WeightedDataFrame(data, columns=cols)
    assert_array_equal(frame.get_weights(), 1)
    assert not frame.isweighted(0) and not frame.isweighted(1)
    assert_array_equal(frame, data)

    frame = WeightedDataFrame(data, weights=None, columns=cols)
    assert_array_equal(frame.get_weights(), 1)
    assert not frame.isweighted(0) and not frame.isweighted(1)
    assert_array_equal(frame, data)

    weights = np.random.rand(N)
    frame = WeightedDataFrame(data, weights=weights, columns=cols)
    assert frame.isweighted(0) and not frame.isweighted(1)
    assert frame.get_weights().shape == (N,)
    assert frame.shape == (N, m)
    assert isinstance(frame.get_weights(), np.ndarray)
    assert_array_equal(frame, data)
    assert_array_equal(frame.get_weights(), weights)
    assert_array_equal(frame.columns, cols)
    assert_array_equal(frame.index.get_level_values('weights'), weights)
    return frame


def test_WeightedDataFrame_key(frame):
    for key1 in frame.columns:
        assert_array_equal(frame.get_weights(), frame[key1].get_weights())
        for key2 in frame.columns:
            assert_array_equal(frame[key1].get_weights(),
                               frame[key2].get_weights())


def test_WeightedDataFrame_slice(frame):
    assert isinstance(frame['A'], WeightedSeries)
    assert isinstance(frame.iloc[0], WeightedSeries)
    assert not frame.iloc[0].isweighted()
    assert isinstance(frame[:10], WeightedDataFrame)
    assert frame[:10].isweighted()
    assert frame[:10].shape == (10, 6)
    assert frame[:10].get_weights().shape == (10,)
    assert frame[:10]._rand(0).shape == (10,)
    assert frame[:10]._rand(1).shape == (6,)


def test_WeightedDataFrame_mean(frame):
    mean = frame.mean()
    assert isinstance(mean, WeightedSeries)
    assert not mean.isweighted()
    assert_array_equal(mean.index, frame.columns)
    assert_allclose(mean, 0.5, atol=1e-2)

    mean = frame.mean(axis=1)
    assert isinstance(mean, WeightedSeries)
    assert_array_equal(mean, frame.T.mean())
    assert isinstance(frame.T.mean(), WeightedSeries)
    assert mean.isweighted()
    assert frame.T.mean().isweighted()
    assert_array_equal(mean.index, frame.index)
    assert_allclose(mean.mean(), 0.5, atol=1e-2)


def test_WeightedDataFrame_std(frame):
    std = frame.std()
    assert isinstance(std, WeightedSeries)
    assert not std.isweighted()
    assert_array_equal(std.index, frame.columns)
    assert_allclose(std, (1./12)**0.5, atol=1e-2)

    std = frame.std(axis=1)
    assert isinstance(std, WeightedSeries)
    assert_array_equal(std, frame.T.std())
    assert isinstance(frame.T.std(), WeightedSeries)
    assert std.isweighted()
    assert frame.T.std().isweighted()
    assert_array_equal(std.index, frame.index)
    assert_allclose(std.mean(), (1./12)**0.5, atol=1e-1)


def test_WeightedDataFrame_cov(frame):
    cov = frame.cov()
    assert isinstance(cov, WeightedDataFrame)
    assert not cov.isweighted(0) and not cov.isweighted(1)
    assert_array_equal(cov.index, frame.columns)
    assert_array_equal(cov.columns, frame.columns)
    assert_allclose(cov, (1./12)*np.identity(6), atol=1e-2)

    cov = frame[:5].T.cov()
    assert isinstance(cov, WeightedDataFrame)
    assert cov.isweighted(0) and cov.isweighted(1)
    assert_array_equal(cov.index, frame[:5].index)
    assert_array_equal(cov.columns, frame[:5].index)

    with pytest.raises(NotImplementedError):
        # kwargs not passed when weighted
        frame.cov(ddof=1)


def test_WeightedDataFrame_corr(frame):
    corr = frame.corr()
    assert isinstance(corr, WeightedDataFrame)
    assert not corr.isweighted(0) and not corr.isweighted(1)
    assert_array_equal(corr.index, frame.columns)
    assert_array_equal(corr.columns, frame.columns)
    assert_allclose(corr, np.identity(6), atol=1.1e-2)

    corr = frame[:5].T.corr()
    assert isinstance(corr, WeightedDataFrame)
    assert corr.isweighted(0) and corr.isweighted(1)
    assert_array_equal(corr.index, frame[:5].index)
    assert_array_equal(corr.columns, frame[:5].index)


def test_WeightedDataFrame_corrwith(frame):
    correl = frame.corrwith(frame.A)
    assert isinstance(correl, WeightedSeries)
    assert not correl.isweighted()
    assert_array_equal(correl.index, frame.columns)
    assert_allclose(correl, frame.corr()['A'], atol=1e-2)

    correl = frame.corrwith(frame[['A', 'B']])
    assert isinstance(correl, WeightedSeries)
    assert not correl.isweighted()
    assert_array_equal(correl.index, frame.columns)
    assert_allclose(correl['A'], 1, atol=1e-2)
    assert_allclose(correl['B'], 1, atol=1e-2)
    assert np.isnan(correl['C'])

    unweighted = DataFrame(frame).droplevel('weights')

    with pytest.raises(ValueError):
        frame.corrwith(unweighted.A)

    with pytest.raises(ValueError):
        frame.corrwith(unweighted[['A', 'B']])

    with pytest.raises(ValueError):
        unweighted.corrwith(frame[['A', 'B']])

    correl_1 = unweighted[:5].corrwith(unweighted[:4], axis=1)
    correl_2 = frame[:5].corrwith(frame[:4], axis=1)
    assert_array_equal(correl_1.values, correl_2.values)
    assert correl_2.isweighted()

    correl_3 = frame[:5].T.corrwith(frame[:4].T)
    assert_array_equal(correl_2, correl_3)
    assert_array_equal(correl_2.index, correl_3.index)

    correl_4 = frame.T.corrwith(frame.T, axis=1)
    correl_5 = unweighted.T.corrwith(unweighted.T, axis=1)
    assert_allclose(correl_4, correl_5)

    frame.set_weights(None, inplace=True)
    assert_array_equal(frame.corrwith(frame), unweighted.corrwith(unweighted))


def test_WeightedDataFrame_median(frame):
    median = frame.median()
    assert isinstance(median, WeightedSeries)
    assert not median.isweighted()
    assert_array_equal(median.index, frame.columns)
    assert_allclose(median, 0.5, atol=1e-2)

    median = frame.median(axis=1)
    assert isinstance(median, WeightedSeries)
    assert_array_equal(median, frame.T.median())
    assert isinstance(frame.T.median(), WeightedSeries)
    assert median.isweighted()
    assert frame.T.median().isweighted()
    assert_array_equal(median.index, frame.index)
    assert_allclose(median.mean(), 0.5, atol=1e-2)


def test_WeightedDataFrame_sem(frame):
    sem = frame.sem()
    assert isinstance(sem, WeightedSeries)
    assert not sem.isweighted()
    assert_array_equal(sem.index, frame.columns)
    assert_allclose(sem, (1./12)**0.5/np.sqrt(frame.neff()), atol=1e-2)

    sem = frame.sem(axis=1)
    assert isinstance(sem, WeightedSeries)
    assert_array_equal(sem, frame.T.sem())
    assert isinstance(frame.T.sem(), WeightedSeries)
    assert sem.isweighted()
    assert frame.T.sem().isweighted()
    assert_array_equal(sem.index, frame.index)


def test_WeightedDataFrame_kurtosis(frame):
    kurtosis = frame.kurtosis()
    assert isinstance(kurtosis, WeightedSeries)
    assert not kurtosis.isweighted()
    assert_array_equal(kurtosis.index, frame.columns)
    assert_allclose(kurtosis, 9./5, atol=1e-2)
    assert_array_equal(frame.kurtosis(), frame.kurt())

    kurtosis = frame.kurtosis(axis=1)
    assert isinstance(kurtosis, WeightedSeries)
    assert_array_equal(kurtosis, frame.T.kurtosis())
    assert isinstance(frame.T.kurtosis(), WeightedSeries)
    assert kurtosis.isweighted()
    assert frame.T.kurtosis().isweighted()
    assert_array_equal(kurtosis.index, frame.index)
    assert_array_equal(frame.kurtosis(axis=1), frame.kurt(axis=1))


def test_WeightedDataFrame_skew(frame):
    skew = frame.skew()
    assert isinstance(skew, WeightedSeries)
    assert not skew.isweighted()
    assert_array_equal(skew.index, frame.columns)
    assert_allclose(skew, 0., atol=2e-2)

    skew = frame.skew(axis=1)
    assert isinstance(skew, WeightedSeries)
    assert_array_equal(skew, frame.T.skew())
    assert isinstance(frame.T.skew(), WeightedSeries)
    assert skew.isweighted()
    assert frame.T.skew().isweighted()
    assert_array_equal(skew.index, frame.index)


def test_WeightedDataFrame_mad(frame):
    mad = frame.mad()
    assert isinstance(mad, WeightedSeries)
    assert not mad.isweighted()
    assert_array_equal(mad.index, frame.columns)
    assert_allclose(mad, 0.25, atol=1e-2)

    mad = frame.mad(axis=1)
    assert isinstance(mad, WeightedSeries)
    assert_array_equal(mad, frame.T.mad())
    assert isinstance(frame.T.mad(), WeightedSeries)
    assert mad.isweighted()
    assert frame.T.mad().isweighted()
    assert_array_equal(mad.index, frame.index)


def test_WeightedDataFrame_quantile(frame):
    quantile = frame.quantile()
    assert isinstance(quantile, WeightedSeries)
    assert not quantile.isweighted()
    assert_array_equal(quantile.index, frame.columns)
    assert_allclose(quantile, 0.5, atol=1e-2)

    quantile = frame.quantile(axis=1)
    assert isinstance(quantile, WeightedSeries)
    assert_array_equal(quantile, frame.T.quantile())
    assert quantile.isweighted()
    assert frame.T.quantile().isweighted()
    assert_array_equal(quantile.index, frame.index)
    assert_allclose(quantile.mean(), 0.5, atol=1e-2)

    qs = np.linspace(0, 1, 10)
    for q in qs:
        quantile = frame.quantile(q)
        assert isinstance(quantile, WeightedSeries)
        assert not quantile.isweighted()
        assert_array_equal(quantile.index, frame.columns)
        assert_allclose(quantile, q, atol=1e-2)

        quantile = frame.quantile(q, axis=1)
        assert isinstance(quantile, WeightedSeries)

        quantile = frame.quantile(q, axis=1)
        assert isinstance(quantile, WeightedSeries)
        assert_array_equal(quantile, frame.T.quantile(q))
        assert quantile.isweighted()
        assert frame.T.quantile(q).isweighted()
        assert_array_equal(quantile.index, frame.index)

    quantile = frame.quantile(qs)
    assert_allclose(quantile, np.transpose([qs]*6), atol=1e-2)
    assert_array_equal(quantile.index, qs)
    assert_array_equal(quantile.columns, frame.columns)
    quantile = frame.quantile(qs, axis=1)
    assert isinstance(quantile, WeightedDataFrame)
    assert_array_equal(quantile.index, qs)
    assert_array_equal(quantile.columns, frame.index)

    with pytest.raises(NotImplementedError):
        frame.quantile(numeric_only=False)
    with pytest.raises(NotImplementedError):
        frame.quantile(method='single')


def test_WeightedDataFrame_sample(frame):
    sample = frame.sample()
    assert isinstance(sample, WeightedDataFrame)
    assert_array_equal(sample.columns, frame.columns)
    assert sample.index.isin(frame.index)
    assert sample.isin(frame).all().all()

    samples = frame.sample(5)
    assert isinstance(samples, WeightedDataFrame)
    assert_array_equal(sample.columns, frame.columns)
    assert samples.index.isin(frame.index).all()
    assert samples.isin(frame).all().all()
    assert len(samples) == 5

    sample = frame.sample(axis=1)
    assert isinstance(sample, WeightedDataFrame)
    assert_array_equal(sample.index, frame.index)
    assert sample.columns.isin(frame.columns).all()
    assert sample.isin(frame).all().all()

    samples = frame.sample(5, axis=1)
    assert isinstance(samples, WeightedDataFrame)
    assert_array_equal(samples.index, frame.index)
    assert samples.columns.isin(frame.columns).all()
    assert samples.isin(frame).all().all()
    assert len(samples.columns) == 5

    frame.T.sample()
    frame.T.sample(5)
    frame.T.sample(axis=1)
    frame.T.sample(5, axis=1)


def test_WeightedDataFrame_neff(frame):
    N_eff = frame.neff()
    assert isinstance(N_eff, float)
    assert N_eff < len(frame)
    assert N_eff > len(frame) * np.exp(-0.25)

    N_eff = frame.neff(1)
    assert isinstance(N_eff, int)
    assert N_eff == len(frame.T)

    # beta kwarg
    for beta in [0.5, 1, 2, np.inf, '0.5', 'equal', 'entropy', 'kish']:
        assert frame.neff(beta=beta) == neff(frame.get_weights(), beta=beta)


def test_WeightedDataFrame_compress(frame):
    assert_allclose(frame.neff(), len(frame.compress()), rtol=1e-2)
    for i in np.logspace(3, 5, 10):
        assert_allclose(i, len(frame.compress(i)), rtol=1e-1)
    unit_weights = frame.compress('equal')
    assert len(np.unique(unit_weights.index)) == len(unit_weights)
    assert_array_equal(frame.compress(), frame.compress())
    assert_array_equal(frame.compress(i), frame.compress(i))
    assert_array_equal(frame.compress('equal'), frame.compress('equal'))

    assert_array_equal(frame.T.compress().T, frame)
    assert_array_equal(frame.T.compress(axis=1).T, frame.compress())


def test_WeightedDataFrame_nan(frame):
    frame['A'][0] = np.nan
    assert ~frame.mean().isna().any()
    assert ~frame.mean(axis=1).isna().any()
    assert_array_equal(frame.mean(skipna=False).isna(), [True] + [False]*5)
    assert_array_equal(frame.mean(axis=1, skipna=False).isna()[0:6],
                       [True, False, False, False, False, False])

    assert ~frame.std().isna().any()
    assert ~frame.std(axis=1).isna().any()
    assert_array_equal(frame.std(skipna=False).isna(), [True] + [False]*5)
    assert_array_equal(frame.std(axis=1, skipna=False).isna()[0:6],
                       [True, False, False, False, False, False])

    frame['B'][2] = np.nan
    assert ~frame.mean().isna().any()
    assert_array_equal(frame.mean(skipna=False).isna(),
                       [True, True] + [False]*4)
    assert_array_equal(frame.mean(axis=1, skipna=False).isna()[0:6],
                       [True, False, True, False, False, False])

    assert ~frame.std().isna().any()
    assert_array_equal(frame.std(skipna=False).isna(),
                       [True, True] + [False]*4)
    assert_array_equal(frame.std(axis=1, skipna=False).isna()[0:6],
                       [True, False, True, False, False, False])

    frame['C'][4] = np.nan
    frame['D'][5] = np.nan
    frame['E'][6] = np.nan
    frame['F'][7] = np.nan
    assert ~frame.mean().isna().any()
    assert frame.mean(skipna=False).isna().all()
    assert_array_equal(frame.mean(axis=1, skipna=False).isna()[0:6],
                       [True, False, True, False, True, True])

    assert ~frame.std().isna().any()
    assert frame.std(skipna=False).isna().all()
    assert_array_equal(frame.std(axis=1, skipna=False).isna()[0:6],
                       [True, False, True, False, True, True])

    assert_allclose(frame.mean(), 0.5, atol=1e-2)
    assert_allclose(frame.std(), (1./12)**0.5, atol=1e-2)
    assert_allclose(frame.cov(), (1./12)*np.identity(6), atol=1e-2)

    assert isinstance(frame.mean(), WeightedSeries)
    assert not frame.mean().isweighted()
    assert isinstance(frame.mean(axis=1), WeightedSeries)
    assert frame.mean(axis=1).isweighted()

    assert frame[:0].mean().isna().all()
    assert frame[:0].std().isna().all()
    assert frame[:0].median().isna().all()
    assert frame[:0].var().isna().all()
    assert frame[:0].cov().isna().all().all()
    assert frame[:0].corr().isna().all().all()
    assert frame[:0].kurt().isna().all()
    assert frame[:0].skew().isna().all()
    assert frame[:0].mad().isna().all()
    assert frame[:0].sem().isna().all()
    assert frame[:0].quantile().isna().all()


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

    frame['A'][0] = np.nan
    assert_allclose(frame.A.cov(frame.A), 1./12, atol=1e-2)
    assert_allclose(frame.A.cov(frame.B), 0, atol=1e-2)


def test_WeightedSeries_corr(frame):
    assert_allclose(frame.A.corr(frame.A), 1., atol=1e-2)
    assert_allclose(frame.A.corr(frame.B), 0, atol=1e-2)
    D = frame.A + frame.B
    assert_allclose(frame.A.corr(D), 1/np.sqrt(2), atol=1e-2)

    unweighted = DataFrame(frame).droplevel('weights')

    with pytest.raises(ValueError):
        frame.A.corr(unweighted.B)

    with pytest.raises(ValueError):
        unweighted.A.corr(frame.B)


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
    unit_weights = series.compress('equal')
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

    assert np.isnan(series[:0].mean())
    assert np.isnan(series[:0].std())
    assert np.isnan(series[:0].median())
    assert np.isnan(series[:0].var())
    assert np.isnan(series[:0].cov(series))
    assert np.isnan(series[:0].corr(series))
    assert np.isnan(series[:0].kurt())
    assert np.isnan(series[:0].skew())
    assert np.isnan(series[:0].mad())
    assert np.isnan(series[:0].sem())
    assert np.isnan(series[:0].quantile())


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


@pytest.fixture
def df_small():
    np.random.seed(42)
    ncol = 3
    ndat = 10
    dat = np.random.normal(loc=5, scale=1, size=(ndat, ncol))
    return DataFrame(dat, columns=["x", "y", "z"])


@pytest.fixture
def wdf_small(df_small):
    np.random.seed(42)
    w = np.random.rand(df_small.shape[0])
    return WeightedDataFrame(df_small.to_numpy(), weights=w,
                             columns=df_small.columns)


def test_WeightedDataFrame_hist(mcmc_df, mcmc_wdf):
    df_axes = mcmc_df.hist().flatten()
    wdf_axes = mcmc_wdf.hist().flatten()

    df_heights = [p.get_height() for ax in df_axes for p in ax.patches]
    wdf_heights = [p.get_height() for ax in wdf_axes for p in ax.patches]
    assert df_heights == wdf_heights

    df_widths = [p.get_width() for ax in df_axes for p in ax.patches]
    wdf_widths = [p.get_width() for ax in wdf_axes for p in ax.patches]
    assert df_widths == wdf_widths

    df_xys = [p.get_xy() for ax in df_axes for p in ax.patches]
    wdf_xys = [p.get_xy() for ax in wdf_axes for p in ax.patches]
    assert df_xys == wdf_xys

    df_axes = mcmc_df.plot.hist(subplots=True)
    wdf_axes = mcmc_wdf.plot.hist(subplots=True)

    df_heights = [p.get_height() for ax in df_axes for p in ax.patches]
    wdf_heights = [p.get_height() for ax in wdf_axes for p in ax.patches]
    assert df_heights == wdf_heights

    df_widths = [p.get_width() for ax in df_axes for p in ax.patches]
    wdf_widths = [p.get_width() for ax in wdf_axes for p in ax.patches]
    assert df_widths == wdf_widths

    df_xys = [p.get_xy() for ax in df_axes for p in ax.patches]
    wdf_xys = [p.get_xy() for ax in wdf_axes for p in ax.patches]
    assert df_xys == wdf_xys


def test_WeightedSeries_hist(mcmc_df, mcmc_wdf):

    fig, axes = plt.subplots(2)
    mcmc_df.x.hist(ax=axes[0])
    mcmc_wdf.x.hist(ax=axes[1])

    df_heights = [p.get_height() for p in axes[0].patches]
    wdf_heights = [p.get_height() for p in axes[1].patches]
    assert df_heights == wdf_heights

    df_widths = [p.get_width() for p in axes[0].patches]
    wdf_widths = [p.get_width() for p in axes[1].patches]
    assert df_widths == wdf_widths

    df_xys = [p.get_xy() for p in axes[0].patches]
    wdf_xys = [p.get_xy() for p in axes[1].patches]
    assert df_xys == wdf_xys

    fig, axes = plt.subplots(2)
    mcmc_df.x.plot.hist(ax=axes[0])
    mcmc_wdf.x.plot.hist(ax=axes[1])

    df_heights = [p.get_height() for p in axes[0].patches]
    wdf_heights = [p.get_height() for p in axes[1].patches]
    assert df_heights == wdf_heights

    df_widths = [p.get_width() for p in axes[0].patches]
    wdf_widths = [p.get_width() for p in axes[1].patches]
    assert df_widths == wdf_widths

    df_xys = [p.get_xy() for p in axes[0].patches]
    wdf_xys = [p.get_xy() for p in axes[1].patches]
    assert df_xys == wdf_xys


def test_KdePlot(mcmc_df, mcmc_wdf):
    bw_method = 0.3
    fig, axes = plt.subplots(2)
    mcmc_df.x.plot.kde(bw_method=bw_method, ax=axes[0])
    mcmc_wdf.x.plot.kde(bw_method=bw_method, ax=axes[1])
    df_line, wdf_line = axes[0].lines[0], axes[1].lines[0]
    assert (df_line.get_xdata() == wdf_line.get_xdata()).all()
    assert_allclose(df_line.get_ydata(),  wdf_line.get_ydata(), atol=1e-4)


def test_scatter_matrix(mcmc_df, mcmc_wdf):
    axes = scatter_matrix(mcmc_df)
    data = axes[0, 1].collections[0].get_offsets().data
    axes = orig_scatter_matrix(mcmc_df)
    orig_data = axes[0, 1].collections[0].get_offsets().data

    assert_allclose(data, orig_data)

    axes = scatter_matrix(mcmc_wdf)
    data = axes[0, 1].collections[0].get_offsets().data
    n = len(data)
    n_ = neff(mcmc_wdf.get_weights(), beta='equal')
    assert_allclose(n, n_, atol=np.sqrt(n))

    axes = orig_scatter_matrix(mcmc_wdf)
    orig_data = axes[0, 1].collections[0].get_offsets().data
    n = len(orig_data)
    assert n == len(mcmc_wdf)

    axes = scatter_matrix(mcmc_wdf, ncompress=50)
    n = len(axes[0, 1].collections[0].get_offsets().data)
    assert_allclose(n, 50, atol=np.sqrt(50))


def test_bootstrap_plot(mcmc_df, mcmc_wdf):
    bootstrap_plot(mcmc_wdf.x)
    bootstrap_plot(mcmc_wdf.x, ncompress=500)


def test_BoxPlot(mcmc_df, mcmc_wdf):
    mcmc_df.plot.box()
    mcmc_wdf.plot.box()

    fig, ax = plt.subplots()
    mcmc_df.boxplot(ax=ax)

    fig, ax = plt.subplots()
    mcmc_wdf.boxplot(ax=ax)

    fig, ax = plt.subplots()
    mcmc_df.x.plot.box(ax=ax)

    fig, ax = plt.subplots()
    mcmc_wdf.x.plot.box(ax=ax)

    mcmc_df.plot.box(subplots=True)
    mcmc_wdf.plot.box(subplots=True)

    mcmc_df['split'] = ''
    mcmc_df.loc[:len(mcmc_df)//2, 'split'] = 'A'
    mcmc_df.loc[len(mcmc_df)//2:, 'split'] = 'B'

    mcmc_wdf['split'] = ''
    mcmc_wdf.iloc[:len(mcmc_wdf)//2, -1] = 'A'
    mcmc_wdf.iloc[len(mcmc_wdf)//2:, -1] = 'B'

    mcmc_df.groupby('split').plot.box()
    mcmc_wdf.groupby('split').plot.box()

    for return_type in ['dict', 'both']:
        fig, ax = plt.subplots()
        mcmc_wdf.plot.box(return_type=return_type, ax=ax)
        fig, ax = plt.subplots()
        mcmc_wdf.boxplot(return_type=return_type, ax=ax)

    fig, ax = plt.subplots()
    mcmc_wdf.boxplot(xlabel='xlabel', ax=ax)
    fig, ax = plt.subplots()
    mcmc_wdf.boxplot(ylabel='ylabel', ax=ax)

    fig, ax = plt.subplots()
    mcmc_wdf.boxplot(vert=False, ax=ax)

    fig, ax = plt.subplots()
    mcmc_wdf.boxplot(fontsize=30, ax=ax)


def test_ScatterPlot(mcmc_df, mcmc_wdf):
    mcmc_df.plot.scatter('x', 'y')
    ax = mcmc_wdf.plot.scatter('x', 'y')

    n = len(ax.collections[0].get_offsets().data)
    n_ = neff(mcmc_wdf.get_weights(), beta='equal')
    assert_allclose(n, n_, atol=np.sqrt(n))

    ax = mcmc_wdf.plot.scatter('x', 'y', ncompress=50)
    n = len(ax.collections[0].get_offsets().data)
    assert_allclose(n, 50, atol=np.sqrt(50))


def test_HexBinPlot(mcmc_df, mcmc_wdf):
    df_axes = mcmc_df.plot.hexbin('x', 'y', mincnt=1)
    wdf_axes = mcmc_wdf.plot.hexbin('x', 'y')

    df_data = df_axes.collections[0].get_offsets()
    wdf_data = wdf_axes.collections[0].get_offsets()
    assert_allclose(df_data, wdf_data)

    df_colors = df_axes.collections[0].get_facecolors()
    wdf_colors = wdf_axes.collections[0].get_facecolors()
    assert_allclose(df_colors, wdf_colors)

    plt.close("all")


def test_AreaPlot(mcmc_df, mcmc_wdf):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    axes_df = mcmc_df.plot.area(ax=ax1)
    axes_wdf = mcmc_wdf.plot.area(ax=ax2)

    assert_allclose(axes_df.get_xlim(), axes_wdf.get_xlim(), rtol=1e-3)
    assert_allclose(axes_df.get_ylim(), axes_wdf.get_ylim(), rtol=1e-3)


def test_BarPlot(mcmc_df, mcmc_wdf):
    axes_bar = mcmc_wdf[5:10].plot.bar()
    axes_barh = mcmc_wdf[5:10].plot.barh()
    assert_array_equal(axes_bar.get_xticks(), axes_barh.get_yticks())
    assert_array_equal(axes_bar.get_yticks(), axes_barh.get_xticks())


def test_PiePlot(mcmc_df, mcmc_wdf):
    fig, ax = plt.subplots()
    mcmc_wdf[5:10].x.plot.pie(ax=ax)
    mcmc_wdf[5:10].plot.pie(subplots=True)


def test_LinePlot(mcmc_df, mcmc_wdf):
    fig, ax = plt.subplots()
    df_axes = mcmc_df.x.plot.line(ax=ax)
    assert_array_equal(mcmc_df.index, df_axes.lines[0].get_xdata())
    fig, ax = plt.subplots()
    wdf_axes = mcmc_wdf.x.plot.line(ax=ax)
    assert_array_equal(mcmc_wdf.index.droplevel('weights'),
                       wdf_axes.lines[0].get_xdata())
    wdf_axes = mcmc_wdf.plot.line()
    assert len(wdf_axes.lines) == len(mcmc_wdf.columns)


def test_multiindex(mcmc_wdf):
    np.random.seed(0)
    i1 = np.arange(len(mcmc_wdf.index))
    i2 = np.random.randint(0, len(mcmc_wdf.index), len(mcmc_wdf.index))
    i3 = np.random.randint(0, len(mcmc_wdf.index), len(mcmc_wdf.index))
    index = MultiIndex.from_arrays([i1, i2, i3], names=['A', 'B', 'C'])
    weights = mcmc_wdf.get_weights()
    wdf = WeightedDataFrame(mcmc_wdf.values, weights=weights, index=index)

    assert wdf.index.names == ['A', 'B', 'C', 'weights']
    assert_allclose(np.array([*wdf.index]).T, [i1, i2, i3, weights])

    assert wdf.reset_index().index.names == [None, 'weights']

    assert not np.array_equal(wdf.reset_index().columns, wdf.columns)
    assert_array_equal(wdf.reset_index(drop=True).columns, wdf.columns)

    new = wdf.copy()
    assert not np.array_equal(new.index, wdf.reset_index())
    new.reset_index(inplace=True)
    assert_array_equal(new.index, wdf.reset_index().index)

    wdf_ = wdf.reset_index(level='A')
    assert wdf_.index.names == ['B', 'C', 'weights']
    wdf_ = wdf.reset_index(level=['A', 'C'])
    assert wdf_.index.names == ['B', 'weights']

    assert_array_equal(wdf.get_weights(), weights)
    wdf_ = wdf.reorder_levels(['B', 'C', 'weights', 'A'])
    assert_array_equal(wdf_.get_weights(), weights)
    weights_ = np.random.rand(len(weights))
    assert_array_equal(wdf_.set_weights(weights_).get_weights(), weights_)

    wdf_ = wdf.droplevel('weights').set_weights(weights, level=2)
    assert wdf_.index.names == ['A', 'B', 'weights', 'C']
    assert_array_equal(wdf_.get_weights(), weights)

    wdf_ = wdf.set_weights(weights, level=2)
    assert wdf_.index.names == ['A', 'B', 'weights', 'C']
    assert_array_equal(wdf_.get_weights(), weights)


def test_weight_passing(mcmc_wdf):
    weights = mcmc_wdf.get_weights()
    new_wdf = WeightedDataFrame(mcmc_wdf.copy(), weights=None)
    assert (new_wdf.get_weights() == mcmc_wdf.get_weights()).all()

    np.random.shuffle(weights)
    assert (mcmc_wdf.get_weights() != weights).any()

    new_wdf = WeightedDataFrame(mcmc_wdf.copy(), weights=weights)
    assert_array_equal(new_wdf.get_weights(), weights)


def test_set_weights(mcmc_wdf):
    weights_1 = mcmc_wdf.get_weights()
    weights_2 = np.random.rand(len(mcmc_wdf.index))

    assert_array_equal(mcmc_wdf.set_weights(weights_2).get_weights(),
                       weights_2)
    assert_array_equal(mcmc_wdf.get_weights(), weights_1)
    mcmc_wdf.set_weights(weights_2, inplace=True)
    assert_array_equal(mcmc_wdf.get_weights(), weights_2)

    assert mcmc_wdf.isweighted()
    assert not mcmc_wdf.set_weights(None).isweighted()
    assert_array_equal(mcmc_wdf.set_weights(None).get_weights(), 1)

    mcmc_wdf.set_weights(None, inplace=True)
    assert not mcmc_wdf.isweighted()
    assert mcmc_wdf.set_weights(None) is not mcmc_wdf

    mcmc_id = id(mcmc_wdf)
    mcmc_wdf.set_weights(None, inplace=True)
    assert id(mcmc_wdf) == mcmc_id
    assert not mcmc_wdf.isweighted()


def test_drop_weights(mcmc_wdf):
    assert mcmc_wdf.isweighted()
    noweights = mcmc_wdf.drop_weights()
    assert not noweights.isweighted()
    assert_array_equal(noweights, mcmc_wdf)
    pandas.testing.assert_frame_equal(noweights.drop_weights(), noweights)
    assert noweights.drop_weights() is not noweights


def test_blank_axis_labels(df_small, wdf_small):
    for df in df_small, wdf_small:
        assert df.plot.area().get_xlabel() == ""
        assert df.plot.bar().get_xlabel() == ""
        assert df.plot.barh().get_ylabel() == ""


def test_get_index(wdf_small):
    wdf_small.index = wdf_small.index.rename(('foo', 'weights'))
    assert wdf_small.plot.area().get_xlabel() == "foo"
    assert wdf_small.plot.bar().get_xlabel() == "foo"
    assert wdf_small.plot.barh().get_ylabel() == "foo"
    for plot in [wdf_small.plot.area, wdf_small.plot.bar, wdf_small.plot.barh]:
        assert plot(xlabel="my xlabel").get_xlabel() == "my xlabel"

    idx = wdf_small.index.to_frame().to_numpy().T
    wdf_small.index = MultiIndex.from_arrays([idx[0], idx[0], idx[1]],
                                             names=('foo', 'bar', 'weights'))
    assert wdf_small.plot.bar().get_xlabel() == "foo,bar"
    assert wdf_small.plot.barh().get_ylabel() == "foo,bar"
    wdf_small.index = MultiIndex.from_arrays([idx[0], idx[0], idx[1]],
                                             names=(None, None, 'weights'))
    assert wdf_small.plot.bar().get_xlabel() == ""
    assert wdf_small.plot.barh().get_ylabel() == ""


def test_style(mcmc_wdf):
    style_dict = dict(x='c', y='y', z='m', w='k')
    ax = mcmc_wdf.plot.kde(style=style_dict)
    assert all([line.get_color() == to_rgba(c)
                for line, c in zip(ax.get_lines(), ['c', 'y', 'm', 'k'])])

    with pytest.raises(TypeError):
        ax = mcmc_wdf.plot.hist_2d('x', 'y', style='c')
    with pytest.raises(TypeError):
        ax = mcmc_wdf.plot.kde_2d('x', 'y', style='c')
    with pytest.raises(TypeError):
        ax = mcmc_wdf.plot.scatter_2d('x', 'y', style='c')
