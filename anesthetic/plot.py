import numpy
import matplotlib.pyplot as plt
from fastkde import fastKDE
from anesthetic.kde import kde_1d, kde_2d
from anesthetic.utils import check_bounds
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
        ax.set_xlabel(tex[p])
        ax.set_yticks([])
        ax.set_ylim(0,1.1)
        ax.xaxis.set_major_locator(MaxNLocator(2))

    
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
    
    for y, (p_y, row) in enumerate(zip(paramnames_y, axes)):
        for x, (p_x, ax) in enumerate(zip(paramnames_x, row)):
            # y labels
            if x==0:
                ax.set_ylabel(tex[p_y])
                ax.yaxis.set_major_locator(MaxNLocator(2))
            else:
                ax.tick_params('y',left=False)

            # x labels
            if y==n_y-1:
                ax.set_xlabel(tex[p_x])
                ax.xaxis.set_major_locator(MaxNLocator(2))
            else:
                ax.tick_params('x',bottom=False)

            # 1D plots
            if p_x == p_y:
                axes[y,x] = ax.twinx()
                axes[y,x].set_yticks([])
                axes[y,x].set_ylim(0,1.1)

    return fig, axes


def plot_1d(data, weights, ax=None, colorscheme=None, xmin=None, xmax=None,
            *args, **kwargs):
    if ax is None:
        ax = plt.gca()

    x, p = kde_1d(numpy.repeat(data, weights), xmin, xmax)
    p /= p.max()
    i = (p>=1e-2)

    ans = ax.plot(x[i], p[i], color=colorscheme, linewidth=1, *args, **kwargs)
    ax.set_xlim(*check_bounds(x[i], xmin, xmax), auto=True)
    return ans


def contour_plot_2d(data_x, data_y, weights, ax=None, colorscheme='b',
                    xmin=None, xmax=None, ymin=None, ymax=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()

    x, y, pdf = kde_2d(numpy.repeat(data_x, weights), numpy.repeat(data_y, weights),
                       xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    pdf /= pdf.max()
    p = sorted(pdf.flatten())
    m = numpy.cumsum(p)
    m /= m[-1]
    interp = interp1d([0]+list(m)+[1],[0]+list(p)+[1])
    contours = list(interp([0.05, 0.33]))+[1]
    if min(p) != 0:
        contours = [min(p)] + contours

    i = (pdf>=1e-2).any(axis=0)
    j = (pdf>=1e-2).any(axis=1)

    zorder = max([child.zorder for child in ax.get_children()])

    cbar = ax.contourf(x[i], y[j], pdf[numpy.ix_(j,i)], contours, vmin=0,vmax=1.2, cmap=plt.cm.get_cmap(convert[colorscheme]), zorder=zorder+1, *args, **kwargs)  
    ax.contour(x[i], y[j], pdf[numpy.ix_(j,i)], contours, vmin=0,vmax=1.2, linewidths=0.5, colors='k', zorder=zorder+2, *args, **kwargs)  
    ax.set_xlim(*check_bounds(x[i], xmin, xmax), auto=True)
    ax.set_ylim(*check_bounds(y[j], ymin, ymax), auto=True)
    return cbar


def scatter_plot_2d(data_x, data_y, weights, ax=None, colorscheme=None, 
                    xmin=None, xmax=None, ymin=None, ymax=None, *args, **kwargs):

    if ax is None:
        ax = plt.gca()
    x = numpy.repeat(data_x, weights)
    y = numpy.repeat(data_y, weights) 
    points = ax.plot(x, y, 'o', markersize=1, color=colorscheme, *args, **kwargs)
    ax.set_xlim(*check_bounds(x, xmin, xmax), auto=True)
    ax.set_ylim(*check_bounds(y, ymin, ymax), auto=True)
    return points
