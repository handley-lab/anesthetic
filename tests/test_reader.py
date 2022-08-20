import matplotlib_agg  # noqa: F401
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import matplotlib.pyplot as plt
from anesthetic import MCMCSamples, NestedSamples
from anesthetic import read_chains
from anesthetic.read.polychord import read_polychord
from anesthetic.read.getdist import read_getdist
from anesthetic.read.cobaya import read_cobaya
from anesthetic.read.multinest import read_multinest
try:
    import getdist
except ImportError:
    pass


def test_read_getdist():
    np.random.seed(3)
    mcmc = read_getdist(root='./tests/example_data/gd')
    assert isinstance(mcmc, MCMCSamples)
    w = np.concatenate((
        np.loadtxt("./tests/example_data/gd_1.txt", usecols=0),
        np.loadtxt("./tests/example_data/gd_2.txt", usecols=0)
    ))
    assert_array_equal(mcmc.get_weights(), w)
    params = ['x0', 'x1', 'x2', 'x3', 'x4', 'logL', 'chain']
    assert_array_equal(mcmc.columns, params)
    tex = {'x0': '$x_0$', 'x1': '$x_1$', 'x2': '$x_2$', 'x3': '$x_3$',
           'x4': '$x_4$', 'chain': r'$n_\mathrm{chain}$'}
    assert mcmc.tex == tex
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])

    mcmc = read_getdist('./tests/example_data/gd_single')
    w = np.loadtxt("./tests/example_data/gd_single.txt", usecols=0)
    assert_array_equal(mcmc.get_weights(), w)
    params.remove('chain')
    assert_array_equal(mcmc.columns, params)
    tex.pop('chain')
    assert mcmc.tex == tex
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")

    mcmc = read_getdist('./tests/example_data/gd_no_paramnames')

    params = [0, 1, 2, 3, 4, 'logL']
    assert all(mcmc.columns == params)
    tex = {}
    assert mcmc.tex == tex


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
    assert_array_equal(mcmc.columns, params)
    if 'getdist' in sys.modules:
        tex = {'x0': '$x0$', 'x1': '$x1$', 'chi2': r'$\chi^2$',
               'chi2__norm': r'$\chi^2_\mathrm{norm}$',
               'chain': r'$n_\mathrm{chain}$'}
    else:
        tex = {'chain': r'$n_\mathrm{chain}$'}
    assert mcmc.tex == tex

    mcmc.plot_2d(['x0', 'x1'])
    mcmc.plot_1d(['x0', 'x1'])
    plt.close("all")

    # single chain file
    mcmc = read_cobaya('./tests/example_data/cb_single_chain')
    tex.pop('chain')
    assert mcmc.tex == tex
    params.remove('chain')
    assert_array_equal(mcmc.columns, params)
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
    assert_array_equal(mcmc.columns, params)
    tex = {'x0': r'$10^{-2}\omega_{b }$',
           'x1': r'$\omega_{cdm }$',
           'x2': r'$100\theta_{s }$',
           'x3': '$ln10^{10}A_{s }$',
           'n_s': '$n_{s }$',
           'tau_reio': r'$\tau_{reio }$',
           'A_cib_217': '$A_{cib 217 }$',
           'xi_sz_cib': '$xi_{sz cib }$',
           'A_sz': '$A_{sz }$',
           'ps_A_100_100': '$ps_{A 100 100 }$',
           'ps_A_143_143': '$ps_{A 143 143 }$',
           'ps_A_143_217': '$ps_{A 143 217 }$',
           'ps_A_217_217': '$ps_{A 217 217 }$',
           'ksz_norm': '$ksz_{norm }$',
           'gal545_A_100': '$gal545_{A 100 }$',
           'gal545_A_143': '$gal545_{A 143 }$',
           'gal545_A_143_217': '$gal545_{A 143 217 }$',
           'gal545_A_217': '$gal545_{A 217 }$',
           'calib_100T': '$10^{-3}calib_{100T }$',
           'calib_217T': '$10^{-3}calib_{217T }$',
           'A_planck': '$10^{-2}A_{planck }$',
           'z_reio': '$z_{reio }$',
           'Omega_Lambda': r'$\Omega_{\Lambda }$',
           'YHe': '$YHe$',
           'H0': '$H0$',
           'A_s': '$10^{-9}A_{s }$',
           'chain': r'$n_\mathrm{chain}$'}
    assert mcmc.tex == tex
    assert_array_equal(mcmc.get_weights(), w)
    assert isinstance(mcmc, MCMCSamples)
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")


def test_read_multinest():
    np.random.seed(3)
    ns = read_multinest('./tests/example_data/mn')
    params = ['x0', 'x1', 'x2', 'x3', 'x4', 'logL', 'logL_birth', 'nlive']
    assert_array_equal(ns.columns, params)
    tex = {'x0': '$x_0$',
           'x1': '$x_1$',
           'x2': '$x_2$',
           'x3': '$x_3$',
           'x4': '$x_4$',
           'logL': r'$\log\mathcal{L}$',
           'logL_birth': r'$\log\mathcal{L}_\mathrm{birth}$',
           'nlive': r'$n_\mathrm{live}$'}
    assert ns.tex == tex

    assert isinstance(ns, NestedSamples)
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])

    ns = read_multinest('./tests/example_data/mn_old')
    tex.pop('logL_birth')
    params.remove('logL_birth')
    assert_array_equal(ns.columns, params)
    assert ns.tex == tex
    assert isinstance(ns, NestedSamples)
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")


def test_read_polychord():
    np.random.seed(3)
    ns = read_polychord('./tests/example_data/pc')
    assert isinstance(ns, NestedSamples)
    for key1 in ns.columns:
        assert_array_equal(ns.get_weights(), ns[key1].get_weights())
        for key2 in ns.columns:
            assert_array_equal(ns[key1].get_weights(), ns[key2].get_weights())
    params = ['x0', 'x1', 'x2', 'x3', 'x4', 'logL', 'logL_birth', 'nlive']
    assert_array_equal(ns.columns, params)
    tex = {'x0': '$x_0$',
           'x1': '$x_1$',
           'x2': '$x_2$',
           'x3': '$x_3$',
           'x4': '$x_4$',
           'logL': r'$\log\mathcal{L}$',
           'logL_birth': r'$\log\mathcal{L}_\mathrm{birth}$',
           'nlive': r'$n_\mathrm{live}$'}
    assert ns.tex == tex

    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")

    with pytest.warns(UserWarning, match="loadtxt"):
        ns_zero_live = read_polychord('./tests/example_data/pc_zero_live')
    ns_nolive = read_polychord('./tests/example_data/pc_no_live')
    ns_single_live = read_polychord('./tests/example_data/pc_single_live')

    cols = ['x0', 'x1', 'x2', 'x3', 'x4', 'logL', 'logL_birth']
    assert_array_equal(ns_zero_live[cols], ns[cols])
    assert_array_equal(ns_nolive[cols], ns[cols][:ns_nolive.shape[0]])
    assert_array_equal(ns_single_live[cols], ns[cols])


@pytest.mark.xfail('getdist' not in sys.modules,
                   raises=NameError,
                   reason="requires getdist package")
@pytest.mark.parametrize('root', ['gd', 'cb'])
def test_discard_burn_in(root):
    np.random.seed(3)
    mcmc = read_chains('./tests/example_data/' + root, burn_in=0.3)
    assert isinstance(mcmc, MCMCSamples)
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])

    # for 2 chains of length 1000
    mcmc0 = read_chains('./tests/example_data/' + root)
    assert isinstance(mcmc0, MCMCSamples)
    mcmc1 = read_chains(root='./tests/example_data/' + root, burn_in=1000)
    assert isinstance(mcmc1, MCMCSamples)
    for key in ['x0', 'x1', 'x2', 'x3', 'x4']:
        if key in mcmc0:
            assert key in mcmc1
            assert_array_equal(mcmc0[key][1000:2000], mcmc1[key][:1000])
    mcmc1.plot_2d(['x0', 'x1', 'x2', 'x3', 'x4'])
    mcmc1.plot_1d(['x0', 'x1', 'x2', 'x3', 'x4'])

    mcmc1 = read_chains('./tests/example_data/' + root, burn_in=-1000.1)
    assert isinstance(mcmc1, MCMCSamples)
    for key in ['x0', 'x1', 'x2', 'x3', 'x4']:
        if key in mcmc0:
            assert key in mcmc1
            assert_array_equal(mcmc0[key][-1000:], mcmc1[key][1000:])
    mcmc1.plot_2d(['x0', 'x1', 'x2', 'x3', 'x4'])
    mcmc1.plot_1d(['x0', 'x1', 'x2', 'x3', 'x4'])


def test_read_fail():
    with pytest.raises(FileNotFoundError):
        read_chains('./tests/example_data/foo')
