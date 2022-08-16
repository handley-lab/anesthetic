import matplotlib_agg  # noqa: F401
import os
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import matplotlib.pyplot as plt
from anesthetic import MCMCSamples, NestedSamples
from anesthetic import read_chains
from anesthetic.read.polychord import read_polychord
from anesthetic.read.getdist import read_getdist
from anesthetic.read.cobaya import read_cobaya, read_paramnames
from anesthetic.read.multinest import read_multinest
try:
    import getdist
except ImportError:
    pass


def test_read_getdist():
    np.random.seed(3)
    mcmc = read_getdist('./tests/example_data/gd')
    assert isinstance(mcmc, MCMCSamples)
    w = np.concatenate((
        np.loadtxt("./tests/example_data/gd_1.txt", usecols=0),
        np.loadtxt("./tests/example_data/gd_2.txt", usecols=0)
    ))
    assert_array_equal(mcmc.weights, w)
    assert 'chain' in mcmc
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])

    mcmc = read_getdist('./tests/example_data/gd_single')
    assert 'chain' not in mcmc
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")


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
    assert_array_equal(mcmc.weights, w)
    pn = read_paramnames("./tests/example_data/cb")
    pn.append('logL')
    pn.append('chain')
    assert_array_equal(mcmc.columns.to_list(), pn)

    mcmc.plot_2d(['x0', 'x1'])
    mcmc.plot_1d(['x0', 'x1'])
    plt.close("all")

    # single chain file
    mcmc = read_cobaya('./tests/example_data/cb_single_chain')
    assert 'chain' not in mcmc
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
    assert_array_equal(mcmc.weights, w)
    assert isinstance(mcmc, MCMCSamples)
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")


def test_read_multinest():
    np.random.seed(3)
    ns = read_multinest('./tests/example_data/mn')
    assert isinstance(ns, NestedSamples)
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])

    ns = read_multinest('./tests/example_data/mn_old')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")


def test_read_polychord():
    np.random.seed(3)
    ns = read_polychord('./tests/example_data/pc')
    assert isinstance(ns, NestedSamples)
    for key1 in ns.columns:
        assert_array_equal(ns.weights, ns[key1].weights)
        for key2 in ns.columns:
            assert_array_equal(ns[key1].weights, ns[key2].weights)
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")

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
