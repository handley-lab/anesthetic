import anesthetic.examples._matplotlib_agg  # noqa: F401
import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import matplotlib.pyplot as plt
from anesthetic.testing import assert_frame_equal
from anesthetic import MCMCSamples, NestedSamples
from anesthetic import read_chains
from anesthetic.read.polychord import read_polychord
from anesthetic.read.getdist import read_getdist
from anesthetic.read.cobaya import read_cobaya
from anesthetic.read.multinest import read_multinest
from anesthetic.read.ultranest import read_ultranest
from anesthetic.read.nestedfit import read_nestedfit
from anesthetic.read.hdf import HDFStore, read_hdf
from utils import pytables_mark_xfail, h5py_mark_xfail, getdist_mark_skip


@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    yield
    plt.close("all")


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
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])

    mcmc = read_getdist('./tests/example_data/gd_single')
    w = np.loadtxt("./tests/example_data/gd_single.txt", usecols=0)
    assert_array_equal(mcmc.get_weights(), w)
    params.remove('chain')
    assert_array_equal(mcmc.drop_labels().columns, params)
    labels.remove(r'$n_\mathrm{chain}$')
    assert_array_equal(mcmc.get_labels(), labels)
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])

    os.rename('./tests/example_data/gd.paramnames',
              './tests/example_data/gd.paramnames_')
    mcmc = read_getdist('./tests/example_data/gd')
    os.rename('./tests/example_data/gd.paramnames_',
              './tests/example_data/gd.paramnames')

    params = [0, 1, 2, 3, 4, 'logL', 'chain']
    assert all(mcmc.drop_labels().columns == params)
    labels = ['', '', '', '', '', r'$\ln\mathcal{L}$', r'$n_\mathrm{chain}$']
    assert_array_equal(mcmc.get_labels(), labels)


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
              'chi2__norm', 'logL', 'chain']
    assert_array_equal(mcmc.drop_labels().columns, params)

    labels = ['$x_0$', '$x_1$', '', '', r'$\chi^2$',
              r'$\chi^2_\mathrm{norm}$', r'$\ln\mathcal{L}$',
              r'$n_\mathrm{chain}$']

    if getdist_mark_skip.args[0]:
        labels[:6] = [''] * 6

    assert_array_equal(mcmc.get_labels(), labels)

    mcmc.plot_2d(['x0', 'x1'])
    mcmc.plot_1d(['x0', 'x1'])

    # single chain file
    mcmc = read_cobaya('./tests/example_data/cb_single_chain')
    params.remove('chain')
    assert_array_equal(mcmc.drop_labels().columns, params)
    labels.remove(r'$n_\mathrm{chain}$')
    assert_array_equal(mcmc.get_labels(), labels)
    # compare directly with getdist
    if not getdist_mark_skip.args[0]:
        import getdist
        mcmc_gd = getdist.loadMCSamples(
            file_root="./tests/example_data/cb_single_chain"
        )
        assert_array_almost_equal(mcmc.logL, mcmc_gd.loglikes, decimal=15)


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
    params = ['a', 'b', 'c', 'd', 'logL', 'logL_birth', 'nlive']
    assert_array_equal(ns.drop_labels().columns, params)
    labels = ['a',
              'b',
              'c',
              'd',
              r'$\ln\mathcal{L}$',
              r'$\ln\mathcal{L}_\mathrm{birth}$',
              r'$n_\mathrm{live}$']
    assert_array_equal(ns.get_labels(), labels)

    assert isinstance(ns, NestedSamples)
    ns.plot_2d(['a', 'b', 'c', 'd'])
    ns.plot_1d(['a', 'b', 'c', 'd'])


def test_read_nestedfit():
    np.random.seed(3)
    ns = read_nestedfit('./tests/example_data/nf')
    params = ['bg', 'x0', 'amp', 'sigma', 'logL', 'logL_birth', 'nlive']
    assert_array_equal(ns.drop_labels().columns, params)
    labels = ['bg',
              'x0',
              'amp',
              'sigma',
              r'$\ln\mathcal{L}$',
              r'$\ln\mathcal{L}_\mathrm{birth}$',
              r'$n_\mathrm{live}$']
    assert_array_equal(ns.get_labels(), labels)

    assert isinstance(ns, NestedSamples)
    ns.plot_2d(['bg', 'x0', 'amp', 'sigma'])
    ns.plot_1d(['bg', 'x0', 'amp', 'sigma'])


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


@pytest.mark.parametrize('root', ['gd', 'cb'])
def test_discard_burn_in(root):
    with pytest.raises(KeyError):
        read_chains('./tests/example_data/' + root, burn_in=0.3)


def test_read_fail():
    with pytest.raises(FileNotFoundError):
        read_chains('./tests/example_data/foo')


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
