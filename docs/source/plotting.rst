********
Plotting
********

You can download the example data used here from GitHub or alternatively use
your own chains files.

Example data on GitHub:
https://github.com/williamjameshandley/anesthetic/tree/master/tests/example_data


Marginalised posterior plotting
===============================

Import anesthetic and load the samples:

.. plot:: :context: close-figs

    from anesthetic import read_chains, make_1d_axes, make_2d_axes
    samples = read_chains("../../tests/example_data/pc")


We have plotting tools for 1D plots ...
---------------------------------------

.. plot:: :context: close-figs

    samples.plot_1d('x0')

... multiple 1D plots ...
-------------------------

.. plot:: :context: close-figs

    samples.plot_1d(['x0', 'x1', 'x2', 'x3', 'x4'])

... triangle plots ...
----------------------

.. plot:: :context: close-figs

    samples.plot_2d(['x0', 'x1', 'x2'], kinds='kde')

... triangle plots (with the equivalent scatter plot filling up the left hand side) ...
---------------------------------------------------------------------------------------

.. plot:: :context: close-figs

    samples.plot_2d(['x0', 'x1', 'x2'])

... and rectangle plots.
------------------------

.. plot:: :context: close-figs

    samples.plot_2d([['x0', 'x1', 'x2'], ['x3', 'x4']])

Rectangle plots are pretty flexible with what they can do.
----------------------------------------------------------

.. plot:: :context: close-figs

    samples.plot_2d([['x0', 'x1', 'x2'], ['x2', 'x1']])


|

Plotting kinds: KDE, histogram, and more
========================================

Anesthetic allows for different plotting kinds, which can be specified through
the ``kind`` (or ``kinds``) keyword. The currently implemented plotting kinds are
kernel density estimation (KDE) plots (``'kde_1d'`` and ``'kde_2d'``), histograms
(``'hist_1d'`` and ``'hist_2d'``), and scatter plots (``'scatter_2d'``).

KDE
---

The KDE plots make use of :py:class:`scipy.stats.gaussian_kde`, whose keyword argument
``bw_method`` is forwarded on.

.. plot:: :context: close-figs

    fig, axes = make_1d_axes(['x0', 'x1'], figsize=(5, 3))
    samples.plot_1d(axes, kind='kde_1d', label="KDE")
    axes.iloc[0].legend(loc='upper right', bbox_to_anchor=(1, 1))

.. plot:: :context: close-figs

    fig, axes = make_2d_axes(['x0', 'x1', 'x2'], upper=False)
    samples.plot_2d(axes, kinds=dict(diagonal='kde_1d', lower='kde_2d'), label="KDE")
    axes.iloc[-1, 0].legend(loc='upper right', bbox_to_anchor=(len(axes), len(axes)))

By default, the two-dimensional plots draw the 68 and 95 percent levels.
Different levels can be requested via the ``levels`` keyword:
    
.. plot:: :context: close-figs

    fig, axes = make_2d_axes(['x0', 'x1', 'x2'], upper=False)
    samples.plot_2d(axes, kinds='kde', levels=[0.99994, 0.99730, 0.95450, 0.68269])

Histograms
----------

.. plot:: :context: close-figs

    fig, axes = make_1d_axes(['x0', 'x1'], figsize=(5, 3))
    samples.plot_1d(axes, kind='hist_1d', label="Histogram")
    axes.iloc[0].legend(loc='upper right', bbox_to_anchor=(1, 1))

.. plot:: :context: close-figs

    fig, axes = make_2d_axes(['x0', 'x1', 'x2'], upper=False)
    samples.plot_2d(axes, kinds=dict(diagonal='hist_1d', lower='hist_2d'), 
                    lower_kwargs=dict(bins=30),
                    diagonal_kwargs=dict(bins=20), 
                    label="Histogram")
    axes.iloc[-1, 0].legend(loc='upper right', bbox_to_anchor=(len(axes), len(axes)))

Scatter plot
------------

.. plot:: :context: close-figs

    fig, axes = make_2d_axes(['x0', 'x1', 'x2'], diagonal=False, upper=False)
    samples.plot_2d(axes, kinds=dict(lower='scatter_2d'), label="Scatter")
    axes.iloc[-1, 0].legend(loc='upper right', bbox_to_anchor=(len(axes), len(axes)))

More finegrained control
------------------------

.. plot:: :context: close-figs

    fig, axes = make_2d_axes(['x0', 'x1', 'x2', 'x3', 'x4'])
    samples.plot_2d(axes.iloc[0:2], kinds=dict(diagonal='kde_1d',  lower='kde_2d',     upper='hist_2d'))
    samples.plot_2d(axes.iloc[2:4], kinds=dict(diagonal='hist_1d', lower='hist_2d',    upper='hist_2d'), bins=20)
    samples.plot_2d(axes.iloc[4: ], kinds=dict(diagonal='kde_1d',  lower='scatter_2d', upper='scatter_2d'))


|

Vertical lines or truth values
==============================

.. plot:: :context: close-figs

    fig, axes = make_2d_axes(['x0', 'x1', 'x2'])
    samples.plot_2d(axes, label="posterior samples")
    axes.scatter({'x0': 0, 'x1': 0, 'x2': 0}, marker='*', c='r', label="some truth")
    axes.axlines({'x2': 0.3}, ls=':', c='k', label="some threshold")
    axes.iloc[-1,  0].legend(loc='lower center', bbox_to_anchor=(len(axes)/2, len(axes)))

|

Changing the appearance
=======================

Anesthetic tries to follow matplotlib conventions as much as possible, so most
changes to the appearance should be relatively straight forward for those
familiar with matplotlib. In the following we present some examples, which we
think might be useful. Are you wishing for an example that is missing here?
Please feel free to raise an issue on the anesthetic GitHub page:

https://github.com/williamjameshandley/anesthetic/issues.

Colour
------

There are multiple options when it comes to specifying colours. The simplest is
by providing the ``color`` (or short ``c``) keyword argument. For some other
plotting kinds it might be desirable to distinguish between ``facecolor`` and
``edgecolor`` (or  short ``fc`` and ``ec``), e.g. for unfilled contours (see also
below "`Unfilled contours`_"). Yet in other cases you might prefer specifying a
matplotlib colormap through the ``cmap`` keyword.

.. plot:: :context: close-figs

    fig, axes = make_2d_axes(['x0', 'x1', 'x2'])
    samples.plot_2d(axes.iloc[0:1, :], kinds=dict(diagonal='kde_1d', lower='kde_2d', upper='kde_2d'), c='r')
    samples.plot_2d(axes.iloc[1:2, :], kinds=dict(diagonal='kde_1d', lower='kde_2d', upper='kde_2d'), fc='C0', ec='C1')
    samples.plot_2d(axes.iloc[2:3, :], kinds=dict(diagonal='kde_1d', lower='kde_2d', upper='kde_2d'), cmap=plt.cm.viridis_r, levels=[0.99994, 0.997, 0.954, 0.683])

Figure size
-----------

.. plot:: :context: close-figs

    fig, axes = make_2d_axes(['x0', 'x1', 'x2'], figsize=(4, 4))
    samples.plot_2d(axes)

Legends
-------

.. plot:: :context: close-figs

    fig, axes = make_2d_axes(['x0', 'x1', 'x2'])
    samples.plot_2d(axes, label='Posterior')
    axes.iloc[ 0,  0].legend(loc='lower left',   bbox_to_anchor=(0, 1))
    axes.iloc[ 0, -1].legend(loc='lower right',  bbox_to_anchor=(1, 1))
    axes.iloc[-1,  0].legend(loc='lower center', bbox_to_anchor=(len(axes)/2, len(axes)))

Unfilled contours
-----------------

.. plot:: :context: close-figs

    fig, axes = make_2d_axes(['x0', 'x1', 'x2'])
    samples.plot_2d(axes, kinds=dict(diagonal='kde_1d', lower='kde_2d'), fc=None, c='C0')
    samples.plot_2d(axes, kinds=dict(diagonal='kde_1d', upper='kde_2d'), fc=None, ec='C1')

