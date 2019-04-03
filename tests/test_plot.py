import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from anesthetic.plot import make_1D_axes
from numpy.testing import assert_array_equal

def test_make_1D_axes():
    paramnames = ['A', 'B', 'C', 'D', 'E']
    tex = {'A':'tA', 'B':'tB', 'C':'tC', 'D':'tD', 'E':'tE'}

    # Check no optional arguments
    fig, axes = make_1D_axes(paramnames)
    assert_array_equal(axes.index, paramnames)
    for p, ax in axes.iteritems():
        assert(ax.get_xlabel() == p)
    plt.close(fig)

    # Check tex argument
    fig, axes = make_1D_axes(paramnames, tex=tex)
    for t in tex:
        assert(axes[t].get_xlabel() != t)
        assert(axes[t].get_xlabel() == tex[t])
    plt.close(fig)

    # Check fig argument
    fig0 = plt.figure()
    fig1 = plt.figure()
    fig, axes = make_1D_axes(paramnames)
    assert(fig is fig1)
    fig, axes = make_1D_axes(paramnames, fig=fig0)
    assert(fig is fig0)
    plt.close(fig0)
    plt.close(fig1)

    # Check ncols argument
    fig, axes = make_1D_axes(paramnames, ncols=2)
    nrows, ncols = axes[0].get_subplotspec().get_gridspec().get_geometry()
    assert(ncols==2)
    plt.close(fig)

    # Check gridspec argument
    grid = gs.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[3,1])
    g00 = grid[0,0]
    fig, axes = make_1D_axes(paramnames, subplot_spec=g00)
    assert(g00 is axes[0].get_subplotspec().get_topmost_subplotspec())
