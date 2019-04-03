import numpy
import pandas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
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


def make_2D_axes(paramnames, paramnames_y=None, tex=None, fig=None, subplot_spec=None):
    paramnames_x = paramnames
    if paramnames_y is None:
        paramnames_y = paramnames
    if tex is None:
        tex = {p:p for p in paramnames}

    nx = len(paramnames_x)
    ny = len(paramnames_y)

    axes = pandas.DataFrame(index=paramnames_y, columns=paramnames_x, dtype=object)
    axes[:][:] = None

    if subplot_spec is not None:
        grid = gs.GridSpecFromSubplotSpec(ny, nx, hspace=0, wspace=0,
                                          subplot_spec=subplot_spec)
    else:
        grid = gs.GridSpec(ny, nx, hspace=0, wspace=0)

    if fig is None:
        fig = plt.gcf()

    for y, py in enumerate(paramnames_y):
        for x, px in enumerate(paramnames_x):
            sx = axes[px][paramnames_y[0]]
            sy = axes[paramnames_y[0]][py]
            axes[px][py] = ax = fig.add_subplot(grid[y,x], sharex=sx, sharey=sy)
            
            ax.label_outer()
            if y == ny-1:
                ax.set_xlabel(tex[px])
                ax.xaxis.set_major_locator(MaxNLocator(2))
            else:
                ax.tick_params('x',bottom=False)

            if x == 0:
                ax.set_ylabel(tex[py])
                ax.yaxis.set_major_locator(MaxNLocator(2))
            else:
                ax.tick_params('y',left=False)

    for y, py in enumerate(paramnames_y):
        for x, px in enumerate(paramnames_x):
            if px == py:
                ax1 = axes[px][py].twinx() 
                ax1.set_yticks([])
                ax.tick_params('y',left=False)
                ax1.set_ylim(0,1.1)
                axes[px][py] = ax1

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

    # Correct non-zero edges
    if min(p) != 0:
        contours = [min(p)] + contours

    # Correct level sets
    for i in range(1, len(contours)):
        if contours[i-1]==contours[i]:
            for j in range(i):
                contours[j] = contours[j] - 1e-5
            
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
