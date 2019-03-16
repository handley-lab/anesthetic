import numpy
import warnings
import matplotlib.pyplot as plt
from fastkde import fastKDE
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator

convert={'r':'Reds', 'b':'Blues', 'y':'Yellows', 'g':'Greens', 'k':'Greys'}

def make_1D_axes(paramnames, tex=None):
    if tex is None:
        tex = {p:p for p in paramnames}

    n = len(paramnames)
    cols = int(numpy.ceil(numpy.sqrt(n)))
    rows = int(numpy.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, squeeze=False, gridspec_kw={'wspace':0})

    for p, ax in zip(paramnames, axes.flatten()):
        ax.set_xlabel('$%s$' % tex[p])
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(3))

    
    for ax in axes.flatten()[n:]:
        ax.remove()

    return fig, axes


def make_2D_axes(paramnames, paramnames_y=None, tex=None):
    paramnames_x = paramnames
    if paramnames_y is None:
        paramnames_y = paramnames
    if tex is None:
        tex = {p:p for p in paramnames}

    n_x = len(paramnames_x)
    n_y = len(paramnames_y)
    fig, axes = plt.subplots(n_y, n_x, sharex='col', sharey='row', gridspec_kw={'wspace':0, 'hspace':0}, squeeze=False)


    for p_y, ax in zip(paramnames_y, axes[:,0]):
        ax.set_ylabel('$%s$' % tex[p_y])
        ax.yaxis.set_major_locator(MaxNLocator(3))

    for p_x, ax in zip(paramnames_x, axes[-1,:]):
        ax.set_xlabel('$%s$' % tex[p_x])
        ax.xaxis.set_major_locator(MaxNLocator(3))


    # Unshare any 1D axes
    for j, (p_j, row) in enumerate(zip(paramnames, axes)):
        for i, (p_i, ax) in enumerate(zip(paramnames, row)):
            if p_i == p_j:
                axes[i,j] = ax.twinx()
                axes[i,j].set_yticks([])

    return fig, axes


def plot_1d(data, weights, ax=None, colorscheme=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, x = fastKDE.pdf(numpy.repeat(data, weights))
    p /= p.max()
    i = (p>=1e-2)

    return ax.plot(x[i], p[i], color=colorscheme, *args, **kwargs)


def contour_plot_2d(data_x, data_y, weights, ax=None, colorscheme='b', *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdf, (x, y) = fastKDE.pdf(numpy.repeat(data_x, weights), numpy.repeat(data_y, weights))
    p = sorted(pdf.flatten())
    m = numpy.cumsum(p)
    m /= m[-1]
    interp = interp1d([0]+list(p)+[numpy.inf],[0]+list(m)+[1])
    pmf = interp(pdf)

    i = (pmf>=1e-2).any(axis=0)
    j = (pmf>=1e-2).any(axis=1)

    ax.contour(x[i], y[j], pmf[numpy.ix_(j,i)], [0.05, 0.33, 1], vmin=0,vmax=1, linewidths=0.5, colors='k', *args, **kwargs)  
    cbar = ax.contourf(x[i], y[j], pmf[numpy.ix_(j,i)], [0.05, 0.33, 1], vmin=0,vmax=1, cmap=plt.cm.get_cmap(convert[colorscheme]), *args, **kwargs)  
    return cbar


def scatter_plot_2d(data_x, data_y, weights, ax=None, colorscheme=None, n=1000, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    w = weights / weights.max()
    if w.sum() > n:
        w *= n/w.sum()
    i = w > numpy.random.rand(len(w))
    return ax.plot(data_x[i], data_y[i], 'o', markersize=1, color=colorscheme, *args, **kwargs)
