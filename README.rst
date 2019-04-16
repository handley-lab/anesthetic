===========================================
anesthetic: nested sampling visualisation
===========================================
:anesthetic: nested sampling visualisation
:Author: Will Handley
:Version: 0.8.1
:Homepage: https://github.com/williamjameshandley/anesthetic
:Documentation: http://anesthetic.readthedocs.io/

.. image:: https://travis-ci.org/williamjameshandley/anesthetic.svg?branch=master
   :target: https://travis-ci.org/williamjameshandley/anesthetic
   :alt: Build Status
.. image:: https://circleci.com/gh/williamjameshandley/anesthetic.svg?style=svg
   :target: https://circleci.com/gh/williamjameshandley/anesthetic
.. image:: https://codecov.io/gh/williamjameshandley/anesthetic/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/williamjameshandley/anesthetic
   :alt: Test Coverage Status
.. image:: https://badge.fury.io/py/anesthetic.svg
   :target: https://badge.fury.io/py/anesthetic
   :alt: PyPi location
.. image:: https://readthedocs.org/projects/anesthetic/badge/?version=latest
   :target: https://anesthetic.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://zenodo.org/badge/175663535.svg
   :target: https://zenodo.org/badge/latestdoi/175663535
   :alt: Permanent DOI for this release



This project is still in beta phase. It aims to bring together tools for processing nested sampling chains, leveraging the standard python libraries:

- numpy
- scipy
- matplotlib
- pandas

As well as the state-of-the-art KDE tool:

- fastKDE

You can see it in action in the `plot gallery <http://htmlpreview.github.io/?https://github.com/williamjameshandley/cosmo_example/blob/master/demos/demo.html>`__.

Current functionality includes:

- Computation of Bayesian evidences, Kullback-Liebler divergences and Bayesian model dimensionalities.
- Marginalised 1d and 2d plots.
- Dynamic replaying of nested sampling run

This tool was designed primarily for use with nested sampling, although it can be used for normal MCMC chains.

For an interactive view of a nested sampling run, you can use the ``anesthetic`` script.

.. code:: bash

   $ anesthetic <ns file root>


Installation
------------

anesthetic can be installed via pip

.. code:: bash

    pip install anesthetic

Or via the setup.py

.. code:: bash

    git clone https://github.com/williamjameshandley/anesthetic 
    cd anesthetic
    python setup.py install --user

Another posterior plotting tool?
--------------------------------

::
    
    This is my posterior plotter. There are many like it, but this one is mine.

There are several excellent tools for plotting marginalised posteriors:

- `getdist <http://getdist.readthedocs.io/en/latest/intro.html>`__ 
- `corner <https://corner.readthedocs.io/en/latest/>`__
- `MontePython <http://baudren.github.io/montepython.html>`__
- `pygtc <https://pygtc.readthedocs.io/en/latest/>`__

Why create another one? In general, any dedicated user of software will find that there is some functionality that in their use case is lacking, and the designs of previous codes make such extensions challenging. In my case this was:

1. For large numbers of samples, kernel density estimation is slow, or inaccurate. There are now better state-of-the-art kernel density estimators, such as `fastKDE <https://pypi.org/project/fastkde/>`__, which ameliorate many of these difficulties.

2. Existing tools can make it painfully difficult to define new parameters. Take for example the default cosmomc chain, which defines ``omegabh2``, but not ``omegab``. The transformation is easy, since ``omegab = omegabh2/ (H0/100)**2``, but writing this simple transformation in code is not so trivial. anesthetic solves this issue by storing the samples as a pandas array, for which the relevant code for defining the above new parameter would be

.. code:: python

    from anesthetic import MCMCSamples

    samples = MCMCSamples.read(file_root)          # Load the samples

    h = samples['H0']/100                          # Define h
    samples['omegab'] = samples.omegabh2/h**2      # Define omegab
    samples.tex['omegab'] = '$\Omega_b$'           # Label omegab

    samples.plot_1d('omegab')                      # Simple 1D plot
    
3. Many KDE plotting tools have conventions that don't play well with uniformly distributed parameters, which is a pain if you are trying to plot priors along with your posteriors. ``anesthetic`` has a sensible mechanism, by defining the contours by the amount of iso-probability mass they contain, but colouring the fill in relation to the probability density of the contour.

Features
--------

- Both samples and plotting axes are stored as a ``pandas.DataFrame``, which makes for easy access and modification
- No overlapping tick labels in large plots

To Do
-----
- Fix nlive
- better interfaces for plotting multiple samples
- multiple nested sampler input formats (MultiNest, Dynesty, NeuralNest)
- Read multiple MCMC chains
- Legends
- Implement live point reading for multinest
- Histogram
