""" General plotting utilities."""
import numpy
import matplotlib.gridspec as gs


def corner_plot(fig, params, gridspec=None):
    """ Create a corner plot, as exemplified by getdist and corner.py

    For other examples of these plots, see:
    https://getdist.readthedocs.io
    https://corner.readthedocs.io

    Args:
        params (list(str)):
            list of labels for axes (best put in LaTeX)

    Kwargs:
        gridspec (matplotlib.gridspec.GridSpec):
            Specification for location of subplots in terms of
            matplotlib.gridspec.GridSpec.

    Returns:
        axes (dict{(str, str):matplotlib.axes.Axes}):
            Mapping from pairs of parameters to axes for plotting.
            string pair is in x-y order.
    """
    n = len(params)
    if gridspec is None:
        gridspec = gs.GridSpec(n, n)
    else:
        gridspec = gs.GridSpecFromSubplotSpec(n, n, subplot_spec=gridspec)

    # Set up the diagonal axes
    axes = {}

    # Set up the axes
    for j, p_y in enumerate(params):
        for i, p_x in enumerate(params):
            if i == j:
                axes[(p_x, p_y)] = fig.add_subplot(gridspec[j, i])
                axes[(p_x, p_y)].set_ylim(0, 1.1)
            if i < j:
                sx = axes[(p_x, p_x)]
                if i is 0:
                    sy = None
                else:
                    sy = axes[(params[0], p_y)]
                axes[(p_x, p_y)] = fig.add_subplot(gridspec[j, i],
                                                   sharex=sx, sharey=sy)

    # Do the labels
    for p in params:
        axes[(p, params[-1])].set_xlabel(p)
        if p != params[0]:
            axes[(params[0], p)].set_ylabel(p)

    # Remove unecessary ticks on top axis
    for p in params[:1]:
        axes[(p, p)].set_yticks([])
    for ax in axes.values():
        ax.label_outer()

    return axes


def histogram(a, **kwargs):
    """ Produce a histogram for path-based plotting

    This is a cheap histogram. Necessary if one wants to update the histogram
    dynamically, and redrawing and filling is very expensive.

    This has the same arguments and keywords as numpy.histogram, but is
    normalised to 1.
    """
    hist, bin_edges = numpy.histogram(a, **kwargs)
    xpath, ypath = numpy.empty((2, 4*len(hist)))
    ypath[0::4] = ypath[3::4] = 0
    ypath[1::4] = ypath[2::4] = hist
    xpath[0::4] = xpath[1::4] = bin_edges[:-1]
    xpath[2::4] = xpath[3::4] = bin_edges[1:]
    mx = max(ypath)
    if mx:
        ypath /= max(ypath)
    return xpath, ypath
