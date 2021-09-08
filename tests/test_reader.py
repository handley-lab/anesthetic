import matplotlib_agg  # noqa: F401
import os
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import matplotlib.pyplot as plt
from anesthetic import MCMCSamples, NestedSamples
from anesthetic.read.chainreader import ChainReader
try:
    import getdist
    import montepython  # noqa: F401
except ImportError:
    pass


def test_read_chainreader():
    reader = ChainReader('root')
    assert reader.root == 'root'
    assert reader.paramnames() == (None, {})
    assert reader.limits() == {}
    with pytest.raises(NotImplementedError):
        reader.samples()


def test_read_getdist():
    np.random.seed(3)
    mcmc = MCMCSamples(root='./tests/example_data/gd')
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])

    mcmc = MCMCSamples(root='./tests/example_data/gd_single')
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")


@pytest.mark.xfail('getdist' not in sys.modules,
                   raises=NameError,
                   reason="requires getdist package")
def test_read_cobayamcmc():
    np.random.seed(3)
    mcmc = MCMCSamples(root='./tests/example_data/cb')
    mcmc.plot_2d(['x0', 'x1'])
    mcmc.plot_1d(['x0', 'x1'])
    plt.close("all")

    # compare directly with getdist
    mcmc_gd = getdist.loadMCSamples(file_root="./tests/example_data/cb")
    assert_array_almost_equal(mcmc.logL, mcmc_gd.loglikes, decimal=15)


@pytest.mark.xfail('montepython' not in sys.modules,
                   raises=ImportError,
                   reason="requires montepython package")
def test_read_montepython():
    np.random.seed(3)
    mcmc = MCMCSamples(root='./tests/example_data/mp')
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")


def test_read_multinest():
    np.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/mn')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])

    ns = NestedSamples(root='./tests/example_data/mn_old')
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")


def test_read_polychord():
    np.random.seed(3)
    ns = NestedSamples(root='./tests/example_data/pc')
    for key1 in ns.columns:
        assert_array_equal(ns.weights, ns[key1].weights)
        for key2 in ns.columns:
            assert_array_equal(ns[key1].weights, ns[key2].weights)
    ns.plot_2d(['x0', 'x1', 'x2', 'x3'])
    ns.plot_1d(['x0', 'x1', 'x2', 'x3'])
    plt.close("all")

    os.rename('./tests/example_data/pc_phys_live-birth.txt',
              './tests/example_data/pc_phys_live-birth.txt_')
    ns_nolive = NestedSamples(root='./tests/example_data/pc')
    os.rename('./tests/example_data/pc_phys_live-birth.txt_',
              './tests/example_data/pc_phys_live-birth.txt')

    cols = ['x0', 'x1', 'x2', 'x3', 'x4', 'logL', 'logL_birth']
    assert_array_equal(ns_nolive[cols], ns[cols][:ns_nolive.shape[0]])


@pytest.mark.xfail('getdist' not in sys.modules,
                   raises=NameError,
                   reason="requires getdist package")
@pytest.mark.parametrize('root', ['gd', 'cb'])
def test_discard_burn_in(root):
    np.random.seed(3)
    mcmc = MCMCSamples(burn_in=0.3, root='./tests/example_data/' + root)
    mcmc.plot_2d(['x0', 'x1', 'x2', 'x3'])
    mcmc.plot_1d(['x0', 'x1', 'x2', 'x3'])

    # for 2 chains of length 1000
    mcmc0 = MCMCSamples(root='./tests/example_data/' + root)
    mcmc1 = MCMCSamples(burn_in=1000, root='./tests/example_data/' + root)
    for key in ['x0', 'x1', 'x2', 'x3', 'x4']:
        if key in mcmc0:
            assert key in mcmc1
            assert_array_equal(mcmc0[key][1000:2000], mcmc1[key][:1000])
    mcmc1.plot_2d(['x0', 'x1', 'x2', 'x3', 'x4'])
    mcmc1.plot_1d(['x0', 'x1', 'x2', 'x3', 'x4'])


def test_read_fail():
    with pytest.raises(FileNotFoundError):
        MCMCSamples(root='./tests/example_data/foo')
