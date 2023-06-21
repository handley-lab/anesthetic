import anesthetic.examples._matplotlib_agg  # noqa: F401
import os
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from anesthetic.testing import assert_frame_equal
from anesthetic import MCMCSamples, NestedSamples
from anesthetic import read_chains
from anesthetic.read.polychord import read_polychord
from anesthetic.read.getdist import read_getdist
from anesthetic.read.cobaya import read_cobaya
from anesthetic.read.multinest import read_multinest
import pandas._testing as tm
from anesthetic.io import HDFStore, read_hdf
try:
    import getdist
except ImportError:
    pass


@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    tm.close()


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


@pytest.mark.xfail('getdist' not in sys.modules,
                   raises=NameError,
                   reason="requires getdist package")
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
    if 'getdist' in sys.modules:
        labels = ['$x_0$', '$x_1$', '', '', r'$\chi^2$',
                  r'$\chi^2_\mathrm{norm}$', r'$\ln\mathcal{L}$',
                  r'$n_\mathrm{chain}$']
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
@pytest.mark.xfail('tables' not in sys.modules,
                   raises=ImportError,
                   reason="requires tables package")
def test_hdf5(root):
    samples = read_chains('./tests/example_data/' + root)

    with HDFStore('/tmp/test_hdf5.h5') as store:
        store["samples"] = samples

    with HDFStore('/tmp/test_hdf5.h5') as store:
        assert_frame_equal(samples, store["samples"])
        assert type(store["samples"]) == type(samples)

    samples.to_hdf('/tmp/test_hdf5.h5', 'samples')

    with HDFStore('/tmp/test_hdf5.h5') as store:
        assert_frame_equal(samples, store["samples"])
        assert type(store["samples"]) == type(samples)

    samples_1 = read_hdf('/tmp/test_hdf5.h5', 'samples')
    assert_frame_equal(samples_1, samples)
    assert type(samples_1) == type(samples)
