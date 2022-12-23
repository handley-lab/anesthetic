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

KDE
---

.. plot:: :context: close-figs

    fig, axes = make_1d_axes(['x0', 'x1'], figsize=(5, 3))
    samples.plot_1d(axes, kind='kde_1d', label="KDE")
    axes.iloc[0].legend(loc='upper right', bbox_to_anchor=(1, 1))

.. plot:: :context: close-figs

    fig, axes = make_2d_axes(['x0', 'x1', 'x2'], upper=False)
    samples.plot_2d(axes, kinds=dict(diagonal='kde_1d', lower='kde_2d'), label="KDE")
    axes.iloc[-1, 0].legend(loc='upper right', bbox_to_anchor=(len(axes), len(axes)))

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
changes to the appearance should be relatively straight forward. In the
following some examples. Wishing for an example that is missing here? Raise an
issue on the anesthetic GitHub page: 

https://github.com/williamjameshandley/anesthetic/issues.

Colour
------

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

