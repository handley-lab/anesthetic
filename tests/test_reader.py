import anesthetic.examples._matplotlib_agg  # noqa: F401
import os
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import matplotlib.pyplot as plt
from anesthetic.testing import assert_frame_equal
from anesthetic import MCMCSamples, NestedSamples
from anesthetic import read_chains, read_paramnames
from anesthetic.read.polychord import read_polychord
from anesthetic.read.getdist import read_getdist
from anesthetic.read.cobaya import read_cobaya, _count_samples
from anesthetic.read.multinest import read_multinest
from anesthetic.read.ultranest import read_ultranest
from anesthetic.read.nestedfit import read_nestedfit
from anesthetic.read.hdf import HDFStore, read_hdf
from anesthetic.read.csv import read_csv
from utils import pytables_mark_xfail, h5py_mark_xfail, getdist_mark_skip
import io


@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    yield
    plt.close("all")


@pytest.fixture(params=['cobaya', 'getdist'])
def repeats_root(tmp_path, request):
    root = tmp_path / 'repeats'
    if request.param == 'cobaya':
        header = '# weight    minuslogpost    p0    p1    n0    n1    chi2\n'
        chain_files = [root.with_suffix(f'.{i}.txt') for i in [1, 2]]
    else:
        root.with_suffix('.paramnames').write_text(
            'p0      p_0\n'
            'p1      p_1\n'
            'n0      n_0\n'
            'n1      n_1\n'
            'chi2    \\chi^2\n'
        )
        header = ''
        chain_files = [Path(str(root) + f'_{i}.txt') for i in [1, 2]]

    chain_files[0].write_text(
        header + (
            '       2              10     0     0     0    10      20\n'
            '       3              11     0     0     1    11      22\n'
            '       1              12     1     1     2    12      24\n'
            '       4              13     0     0     3    13      26\n'
        )
    )
    chain_files[1].write_text(
        header + (
            '       5              10     0     0     4    14      20\n'
            '       1              11     0     0     5    15      22\n'
        )
    )
    return str(root)


@pytest.mark.parametrize(('root', 'expected'), [
    ('cb', ['x0', 'x1', 'minuslogprior', 'minuslogprior__0',
            'chi2', 'chi2__norm']),
    ('gd', ['x0', 'x1', 'x2', 'x3', 'x4']),
    ('pc', ['x0', 'x1', 'x2', 'x3', 'x4']),
    ('mn', ['x0', 'x1', 'x2', 'x3', 'x4']),
    ('nf', ['x0', 'x1', 'amp', 'sigma']),
    ('un', ['x0', 'x1', 'x2', 'x3']),
])
def test_read_paramnames(root, expected):
    parameters, _ = read_paramnames(f'./tests/example_data/{root}')
    assert parameters == expected


@pytest.mark.parametrize(('suffix', 'nbookkeeping'), [
    ('.txt', 2),             # getdist
    ('_1.txt', 2),           # getdist
    ('.1.txt', 2),           # getdist
    ('_dead-birth.txt', 2),  # polychord
    ('dead-birth.txt', 4),   # multinest
    ('ev.dat', 3),           # multinest
])
def test_read_paramnames_metadata(tmp_path, suffix, nbookkeeping):
    root = tmp_path / 'chain'
    chain_file = Path(f'{root}{suffix}')
    chain_file.write_text('1 2' + nbookkeeping * ' 1' + '\n')

    # without metadata
    with pytest.warns(UserWarning, match='Using integer parameter names'):
        parameters, labels = read_paramnames(root)
    assert parameters == [0, 1]
    assert labels == {}

    # with metadata
    root.with_suffix('.paramnames').write_text('x0 x_0\nx1 x_1\n')
    parameters, labels = read_paramnames(root)
    assert parameters == ['x0', 'x1']
    assert labels == {'x0': '$x_0$', 'x1': '$x_1$'}


def test_read_getdist():
    np.random.seed(3)
    mcmc = read_getdist('./tests/example_data/gd')
    assert isinstance(mcmc, MCMCSamples)
    w = np.concatenate((
        np.loadtxt("./tests/example_data/gd_1.txt", usecols=0),
        np.loadtxt("./tests/example_data/gd_2.txt", usecols=0)
    ))
    assert_array_equal(mcmc.get_weights(), w)
    params = ['x0', 'x1', 'x2', 'x3', 'x4', 'logL', 'chain']
    assert_array_equal(mcmc.drop_labels().columns, params)
    labels = ['$x_0$', '$x_1$', '$x_2$', '$x_3$', '$x_4$',
              r'$\ln\mathcal{L}$', r'$n_\mathrm{chain}$']
    assert_array_equal(mcmc.get_labels(), labels)
    mcmc = mcmc.remove_burn_in(0.5)
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])

    mcmc = read_getdist('./tests/example_data/gd_single')
    w = np.loadtxt("./tests/example_data/gd_single.txt", usecols=0)
    assert_array_equal(mcmc.get_weights(), w)
    assert_array_equal(mcmc.chain, 0)
    assert_array_equal(mcmc.drop_labels().columns, params)
    assert_array_equal(mcmc.get_labels(), labels)
    mcmc = mcmc.remove_burn_in(0.5)
    assert_array_equal(mcmc.chain, 0)
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])

    os.rename('./tests/example_data/gd.paramnames',
              './tests/example_data/gd.paramnames_')
    with pytest.warns(UserWarning, match="Using integer parameter names"):
        mcmc = read_getdist('./tests/example_data/gd')
    os.rename('./tests/example_data/gd.paramnames_',
              './tests/example_data/gd.paramnames')

    params = [0, 1, 2, 3, 4, 'logL', 'chain']
    assert all(mcmc.drop_labels().columns == params)
    labels = ['', '', '', '', '', r'$\ln\mathcal{L}$', r'$n_\mathrm{chain}$']
    assert_array_equal(mcmc.get_labels(), labels)


@pytest.mark.parametrize(('weights', 'dtype'), [
    ([1, 2], np.integer),     # integer multiplicities
    ([1, 0.5], np.floating),  # importance weights
])
def test_read_getdist_weight_dtype(tmp_path, weights, dtype):
    root = tmp_path / 'chain'
    root.with_suffix('.paramnames').write_text('x0 x_0\n')
    root.with_suffix('.txt').write_text(
        ''.join(f'{weight}  0  0\n' for weight in weights)
    )
    samples = read_getdist(str(root))
    assert np.issubdtype(samples.get_weights().dtype, dtype)
    if np.issubdtype(dtype, np.floating):
        with pytest.raises(ValueError, match="integer frequency weights"):
            samples.thin(2)


def test_read_cobayamcmc():
    np.random.seed(3)
    mcmc = read_cobaya('./tests/example_data/cb')
    assert isinstance(mcmc, MCMCSamples)
    w = np.concatenate((
        np.loadtxt("./tests/example_data/cb.1.txt", usecols=0),
        np.loadtxt("./tests/example_data/cb.2.txt", usecols=0)
    ))
    assert_array_equal(mcmc.get_weights(), w)
    params = ['x0', 'x1', 'minuslogprior', 'minuslogprior__0', 'chi2',
              'chi2__norm', 'logL', 'logP', 'chain']
    assert_array_equal(mcmc.drop_labels().columns, params)

    labels = ['$x_0$', '$x_1$', r'$-\ln\pi$', r'$-\ln\pi_\mathrm{0}$',
              r'$\chi^2$', r'$\chi^2_\mathrm{norm}$', r'$\ln\mathcal{L}$',
              r'$\ln\mathcal{P}$', r'$n_\mathrm{chain}$']

    if getdist_mark_skip.args[0]:
        labels[:6] = [''] * 6

    assert_array_equal(mcmc.get_labels(), labels)

    mcmc.plot_2d(['x0', 'x1'])
    mcmc.plot_1d(['x0', 'x1'])

    # single chain file
    mcmc = read_cobaya('./tests/example_data/cb_single_chain')
    assert_array_equal(mcmc.drop_labels().columns, params)
    assert_array_equal(mcmc.get_labels(), labels)
    # compare directly with getdist
    if not getdist_mark_skip.args[0]:
        import getdist
        g = getdist.loadMCSamples(
            file_root="./tests/example_data/cb_single_chain"
        )
        # Note that GetDist's `loglikes` attribute actually corresponds to
        # `minuslogposterior`. Hence, the following slightly confusing asserts.
        assert_array_almost_equal(mcmc.logP, -g.loglikes, decimal=15)
        assert_array_almost_equal(mcmc.logL, -g.getParams().chi2/2, decimal=15)


@pytest.mark.parametrize(('content', 'expected'), [
    ('# header\n', 0),                     # no samples
    ('# a b\n  1 2\n  3 4\n  5 6\n', 3),   # fixed-width rows
    ('# a b\n  1 2\n  3 4\n  5 6', 3),     # missing final \n
    ('# header\n1 2\n3 4 5\n6 7\n', 3),    # variable-width rows
    ('# header\n1 2\n3 4 5\n6 7', 3),      # missing final \n
    ('# a b\n  1 2\n\n# note\n  5 6', 2),  # blank and comment rows
    ('# header\n1 2\n300 400\n', 2),       # fixed-size total, unequal rows
])
def test_count_cobaya_samples(tmp_path, content, expected):
    chain = tmp_path / 'chain.txt'
    chain.write_text(content)
    assert _count_samples(chain) == expected


@pytest.mark.parametrize(('root', 'bookkeeping'), [
    ('cb', ['chi2', 'logL', 'logP', 'chain']),
    ('gd', ['logL', 'chain']),
])
@pytest.mark.parametrize('burn_in', [[500, 1000], 0.5])
@pytest.mark.parametrize('compress_repeats', [False, True])
def test_read_columns_burnin_thin_compress(root, bookkeeping, burn_in,
                                           compress_repeats):
    root = f'./tests/example_data/{root}'
    columns = ['x0', 'x1'] + (['chain'] if compress_repeats else bookkeeping)
    expected = read_chains(root)[columns].remove_burn_in(burn_in).thin(10)
    expected = expected.compress_repeats() if compress_repeats else expected
    expected.reset_index(drop=True, inplace=True)

    selected = read_chains(root, columns=['x0', 'x1'], thin=10,
                           burn_in=burn_in, compress_repeats=compress_repeats)

    assert_frame_equal(selected, expected)


def test_read_mcmc_compress_repeats(repeats_root):
    mc = read_chains(repeats_root, columns=['p0', 'p1'], compress_repeats=True)
    assert_array_equal(mc.drop_labels().columns, ['p0', 'p1', 'chain'])
    assert_array_equal(mc[['p0', 'p1']], [[0, 0], [1, 1], [0, 0], [0, 0]])
    assert_array_equal(mc.chain, [1, 1, 1, 2])
    assert_array_equal(mc.get_weights(), [5, 1, 4, 6])


def test_read_mcmc_compress_all_and_no_columns(repeats_root):
    all_columns = read_chains(repeats_root, compress_repeats=True)
    assert_array_equal(all_columns.drop_labels().columns,
                       ['p0', 'p1', 'n0', 'n1', 'chi2', 'chain'])
    assert len(all_columns) == 6
    assert 'logP' not in all_columns
    assert 'logL' not in all_columns

    no_columns = read_chains(repeats_root, columns=[], compress_repeats=True)
    assert_array_equal(no_columns.drop_labels().columns, ['chain'])
    assert_array_equal(no_columns.chain, [1, 2])
    assert_array_equal(no_columns.get_weights(), [10, 6])


def test_read_mcmc_columns_burnin_thin_compress(repeats_root):
    mc = read_chains(repeats_root, columns=['p0', 'p1'],
                     burn_in=[1, 0], thin=2, compress_repeats=True)
    # Thinning removes the intervening (p0, p1)=(1, 1) row before compression.
    assert_array_equal(mc[['p0', 'p1']], [[0, 0], [0, 0]])
    assert_array_equal(mc.chain, [1, 2])
    assert_array_equal(mc.get_weights(), [4, 3])


@pytest.mark.parametrize(('compress_repeats', 'weights'),
                         [(False, [5, 1]),
                          (True, [6])])
def test_read_mcmc_empty_chain_after_burn_in(repeats_root, compress_repeats,
                                             weights):
    mc = read_chains(repeats_root, columns=['p0', 'p1'], burn_in=[4, 0],
                     compress_repeats=compress_repeats)
    # Burn-in removes every stored row from the first chain.
    assert_array_equal(mc.chain, np.full(len(weights), 2))
    assert_array_equal(mc.get_weights(), weights)


def test_read_mcmc_compressed_and_uncompressed_thinning_match(repeats_root):
    # uncompressed
    u = read_chains(repeats_root, columns=['p0', 'p1']).thin(2)
    u = u[['p0', 'p1', 'chain']]
    # compressed
    c = read_chains(repeats_root, columns=['p0', 'p1'],
                    compress_repeats=True).thin(2)
    # Expand frequency weights before comparison:
    u = np.repeat(u.to_numpy(), u.get_weights(), axis=0)
    c = np.repeat(c.to_numpy(), c.get_weights(), axis=0)
    assert_array_equal(c, u)


def test_read_montepython():
    np.random.seed(3)
    root = './tests/example_data/mp/2019-01-24_200000_'
    mcmc = read_getdist(root)
    w = np.concatenate((
        np.loadtxt(root + '_1.txt', usecols=0),
        np.loadtxt(root + '_2.txt', usecols=0)
    ))
    params = ['x0', 'x1', 'x2', 'x3', 'n_s', 'tau_reio', 'A_cib_217',
              'xi_sz_cib', 'A_sz', 'ps_A_100_100', 'ps_A_143_143',
              'ps_A_143_217', 'ps_A_217_217', 'ksz_norm', 'gal545_A_100',
              'gal545_A_143', 'gal545_A_143_217', 'gal545_A_217', 'calib_100T',
              'calib_217T', 'A_planck', 'z_reio', 'Omega_Lambda', 'YHe', 'H0',
              'A_s', 'logL', 'chain']
    assert_array_equal(mcmc.drop_labels().columns, params)
    labels = [r'$10^{-2}\omega_{b }$',
              r'$\omega_{cdm }$',
              r'$100\theta_{s }$',
              '$ln10^{10}A_{s }$',
              '$n_{s }$',
              r'$\tau_{reio }$',
              '$A_{cib 217 }$',
              '$xi_{sz cib }$',
              '$A_{sz }$',
              '$ps_{A 100 100 }$',
              '$ps_{A 143 143 }$',
              '$ps_{A 143 217 }$',
              '$ps_{A 217 217 }$',
              '$ksz_{norm }$',
              '$gal545_{A 100 }$',
              '$gal545_{A 143 }$',
              '$gal545_{A 143 217 }$',
              '$gal545_{A 217 }$',
              '$10^{-3}calib_{100T }$',
              '$10^{-3}calib_{217T }$',
              '$10^{-2}A_{planck }$',
              '$z_{reio }$',
              r'$\Omega_{\Lambda }$',
              '$YHe$',
              '$H0$',
              '$10^{-9}A_{s }$',
              r'$\ln\mathcal{L}$',
              r'$n_\mathrm{chain}$']
    assert_array_equal(mcmc.get_labels(), labels)
    assert_array_equal(mcmc.get_weights(), w)
    assert isinstance(mcmc, MCMCSamples)
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])


def test_read_multinest():
    np.random.seed(3)
    ns = read_multinest('./tests/example_data/mn')
    params = ['x0', 'x1', 'x2', 'x3', 'x4', 'logL', 'logL_birth', 'nlive']
    assert_array_equal(ns.drop_labels().columns, params)
    labels = ['$x_0$',
              '$x_1$',
              '$x_2$',
              '$x_3$',
              '$x_4$',
              r'$\ln\mathcal{L}$',
              r'$\ln\mathcal{L}_\mathrm{birth}$',
              r'$n_\mathrm{live}$']
    assert_array_equal(ns.get_labels(), labels)

    assert isinstance(ns, NestedSamples)
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])

    ns = read_multinest('./tests/example_data/mn_old')
    params.remove('logL_birth')
    assert_array_equal(ns.drop_labels().columns, params)
    labels.remove(r'$\ln\mathcal{L}_\mathrm{birth}$')
    assert_array_equal(ns.get_labels(), labels)
    assert isinstance(ns, NestedSamples)
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])


@h5py_mark_xfail
def test_read_ultranest():
    np.random.seed(3)
    ns = read_ultranest('./tests/example_data/un')
    params = ['x0', 'x1', 'x2', 'x3', 'logL', 'logL_birth', 'nlive']
    assert_array_equal(ns.drop_labels().columns, params)
    labels = ['x0',
              'x1',
              'x2',
              'x3',
              r'$\ln\mathcal{L}$',
              r'$\ln\mathcal{L}_\mathrm{birth}$',
              r'$n_\mathrm{live}$']
    assert_array_equal(ns.get_labels(), labels)

    assert isinstance(ns, NestedSamples)
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])


def test_read_nestedfit():
    np.random.seed(3)
    ns = read_nestedfit('./tests/example_data/nf')
    params = ['x0', 'x1', 'amp', 'sigma', 'logL', 'logL_birth', 'nlive']
    assert_array_equal(ns.drop_labels().columns, params)
    labels = ['x0',
              'x1',
              'amp',
              'sigma',
              r'$\ln\mathcal{L}$',
              r'$\ln\mathcal{L}_\mathrm{birth}$',
              r'$n_\mathrm{live}$']
    assert_array_equal(ns.get_labels(), labels)

    assert isinstance(ns, NestedSamples)
    ns.plot_2d(['x0', 'x1', 'amp', 'sigma'])
    ns.plot_1d(['x0', 'x1', 'amp', 'sigma'])


def test_read_polychord():
    np.random.seed(3)
    ns = read_polychord('./tests/example_data/pc')
    assert isinstance(ns, NestedSamples)
    for key1 in ns.columns:
        assert_array_equal(ns.get_weights(), ns[key1].get_weights())
        for key2 in ns.columns:
            assert_array_equal(ns[key1].get_weights(), ns[key2].get_weights())
    params = ['x0', 'x1', 'x2', 'x3', 'x4', 'logL', 'logL_birth', 'nlive']
    assert_array_equal(ns.drop_labels().columns, params)
    labels = ['$x_0$',
              '$x_1$',
              '$x_2$',
              '$x_3$',
              '$x_4$',
              r'$\ln\mathcal{L}$',
              r'$\ln\mathcal{L}_\mathrm{birth}$',
              r'$n_\mathrm{live}$']
    assert_array_equal(ns.get_labels(), labels)

    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])

    os.rename('./tests/example_data/pc_phys_live-birth.txt',
              './tests/example_data/pc_phys_live-birth.txt_')
    ns_nolive = read_polychord('./tests/example_data/pc')
    os.rename('./tests/example_data/pc_phys_live-birth.txt_',
              './tests/example_data/pc_phys_live-birth.txt')

    with pytest.warns(UserWarning, match="loadtxt"):
        ns_zero_live = read_polychord('./tests/example_data/pc_zero_live')

    ns_single_live = read_polychord('./tests/example_data/pc_single_live')

    cols = ['x0', 'x1', 'x2', 'x3', 'x4', 'logL', 'logL_birth']
    assert_array_equal(ns_nolive[cols], ns[cols][:ns_nolive.shape[0]])
    assert_array_equal(ns_zero_live[cols], ns[cols])
    assert_array_equal(ns_single_live[cols], ns[cols])


def test_read_blackjax():
    np.random.seed(3)
    bj = read_chains('./tests/example_data/bj')
    assert isinstance(bj, NestedSamples)
    params = ['x0', 'x1', 'x2', 'x3', 'x4', 'logL', 'logL_birth', 'nlive']
    assert_array_equal(bj.drop_labels().columns, params)
    labels = ['$x_0$',
              '$x_1$',
              '$x_2$',
              '$x_3$',
              '$x_4$',
              r'$\ln\mathcal{L}$',
              r'$\ln\mathcal{L}_\mathrm{birth}$',
              r'$n_\mathrm{live}$']
    assert_array_equal(bj.get_labels(), labels)
    assert bj.nlive[0] == 125
    assert np.isnan(bj.logL_birth[0])
    bj.recompute()
    assert bj.nlive[0] == 125
    assert np.isnan(bj.logL_birth[0])


@pytest.mark.parametrize('root', [
    'gd', 'pc', 'mn', 'cb', 'nf',
    pytest.param('un', marks=h5py_mark_xfail),
])
def test_read_labels(root):
    root = f'./tests/example_data/{root}'
    assert read_chains(root).islabelled()
    assert not read_chains(root, labels=None).islabelled()


@pytest.mark.parametrize(('root', 'bookkeeping'), [
    ('cb', ['chi2', 'logL', 'logP', 'chain']),
    ('gd', ['logL', 'chain']),
    ('pc', ['logL', 'logL_birth', 'nlive']),
    ('mn', ['logL', 'logL_birth', 'nlive']),
    ('mn_old', ['logL', 'nlive']),
    ('nf', ['logL', 'logL_birth', 'nlive']),
    pytest.param('un', ['logL', 'logL_birth', 'nlive'], marks=h5py_mark_xfail),
])
@pytest.mark.parametrize(('columns', 'parameters'), [
    ('x0', ['x0']),                    # scalar name
    (0, ['x0']),                       # scalar index
    (np.int64(0), ['x0']),             # scalar numpy index
    (['x0', 'x1'], ['x0', 'x1']),      # names
    ([0, 1], ['x0', 'x1']),            # indices
    (np.int32([0, 1]), ['x0', 'x1']),  # numpy indices
    (slice(0, 2), ['x0', 'x1']),       # slice
    ([], []),                          # empty selection
    (['x1', 'x0'], ['x1', 'x0']),      # reordered names
    ([1, 0], ['x1', 'x0']),            # reordered indices
    (np.int32([1, 0]), ['x1', 'x0']),  # reordered numpy indices
    (['x0', 'x0'], ['x0', 'x0']),      # repeated names
    ([0, 0], ['x0', 'x0']),            # repeated indices
])
def test_read_columns(root, bookkeeping, columns, parameters):
    root = f'./tests/example_data/{root}'
    expected = read_chains(root)[parameters + bookkeeping]
    selected = read_chains(root, columns=columns)
    assert_frame_equal(selected, expected)


@pytest.mark.parametrize('root', ['cb', 'gd', 'pc', 'mn', 'mn_old', 'nf',
                                  pytest.param('un', marks=h5py_mark_xfail)])
@pytest.mark.parametrize('columns', [[-4], [-3, 1, -2], slice(-4, -2)])
def test_read_columns_negative_indexing(root, columns):
    root = f'./tests/example_data/{root}'
    parameters, _ = read_paramnames(root)
    parameters = np.asarray(parameters)[columns]
    expected = read_chains(root, columns=parameters)
    selected = read_chains(root, columns=columns)
    assert_frame_equal(selected, expected)


@pytest.mark.parametrize(('columns', 'error'), [
    (['x0', 'missing'], KeyError),      # unknown name
    ([0, 10], IndexError),              # out-of-range index
    ([True, False, False], TypeError),  # boolean input
    (1.5, TypeError),                   # unsupported scalar
    ([0, 'x1'], TypeError),             # mixed selector types
])
def test_read_columns_invalid(columns, error):
    with pytest.raises(error):
        read_chains('./tests/example_data/cb', columns=columns)


@pytest.mark.parametrize('read', [read_chains, read_paramnames])
def test_read_fail(read):
    with pytest.raises(FileNotFoundError):
        read('./tests/example_data/foo')


def test_regex_escape():
    mcmc_1 = read_chains('./tests/example_data/gd_single+X')
    mcmc_2 = read_chains('./tests/example_data/gd_single')
    assert_frame_equal(mcmc_1, mcmc_2, check_metadata=False)


@pytest.mark.parametrize('root', ['pc', 'gd'])
@pytables_mark_xfail
def test_hdf5(tmp_path, root):
    samples = read_chains('./tests/example_data/' + root)
    filename = tmp_path / ('test_hdf5' + root + '.h5')
    key = "samples"

    with HDFStore(filename) as store:
        store[key] = samples

    with HDFStore(filename) as store:
        assert_frame_equal(samples, store[key])
        assert type(store[key]) is type(samples)

    samples.to_hdf(filename, key)

    with HDFStore(filename) as store:
        assert_frame_equal(samples, store[key])
        assert type(store[key]) is type(samples)

    samples_ = read_hdf(filename, key)
    assert_frame_equal(samples_, samples)
    assert type(samples_) is type(samples)


@pytest.mark.parametrize('root', ['pc', 'gd'])
def test_path(root):
    base_dir = Path("./tests/example_data")
    read_chains(base_dir / root)


@pytest.mark.parametrize('root', ['pc', 'gd'])
def test_read_csv(tmp_path, root):
    filename = tmp_path / f'{root}.csv'
    samples = read_chains(f'./tests/example_data/{root}')
    samples.to_csv(filename)

    samples_ = read_csv(filename)
    samples_.root = samples.root
    assert_frame_equal(samples, samples_)

    samples_ = read_chains(filename)
    samples_.root = samples.root
    assert_frame_equal(samples, samples_)

    with open(filename, 'rb') as f:
        csv_bytes = f.read()
    bytesio_obj = io.BytesIO(csv_bytes)
    samples_bytes = read_csv(bytesio_obj)
    samples_bytes.root = samples.root
    samples_bytes.label = samples.label
    assert_frame_equal(samples, samples_bytes)
