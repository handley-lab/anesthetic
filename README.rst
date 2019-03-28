===========================================
aNESThetic: nested sampling post-processing 
===========================================
:aNESThetic: nested sampling post-processing
:Author: Will Handley
:Version: 0.5.0
:Homepage: https://github.com/williamjameshandley/anesthetic

This project is still in alpha stage. It aims to bring together tools for processing nested sampling chains, leveraging the standard python libraries:

- numpy
- scipy
- pandas

As well as the state-of-the-art KDE tool:

- fastKDE

You can see it in action in the `plot gallery <http://htmlpreview.github.io/?https://github.com/williamjameshandley/cosmo_example/blob/master/demos/demo.html>`__.

Current functionality includes:
- Computation of Bayesian evidences, Kullback-Liebler divergences and Bayesian model dimensionalities.
- Marginalised 1d and 2d plots.

This tool was designed primarily for use with nested sampling, although it can be used for normal MCMC chains.

Another posterior plotting tool?
--------------------------------

::
    
    This is my posterior plotter. There are many like it, but this one is mine.

There are several excellent tools for plotting marginalised posteriors:

- `getdist <http://getdist.readthedocs.io/en/latest/intro.html>`__ 
- `corner <https://corner.readthedocs.io/en/latest/>`__
- `MontePython <http://baudren.github.io/montepython.html>`__
- `pygtc <https://pygtc.readthedocs.io/en/latest/>`__

Why create another one? In general, any dedicated user of software will find that there is some functionality that in their use case is lacking, and the designs of previous codes make such an extensions challenging. In my case this was:

1. For large numbers of samples, kernel density estimation is slow, or inaccurate. There are now better state-of-the-art kernel density estimators, such as `fastKDE <https://pypi.org/project/fastkde/>`__, which ameliorate many of these difficulties.

2. Existing tools can make it painfully difficult to define new parameters. Take for example the default cosmomc chain, which defines ``omegabh2``, but not ``omegab``. The transformation is easy, since ``omegab = omegabh2/ (H0/100)**2``, but writing this simple transformation in code is not so trivial. anesthetic solves this issue by storing the samples as a pandas array, for which the relevant code for defining the above new parameter would be

.. code:: python

    from anesthetic.anesthetic import MCMCSamples

    samples = MCMCSamples.read(file_root)          # Load the samples

    h = samples['H0']/100                          # Define h
    samples['omegab'] = samples.omegabh2/h**2      # Define omegab
    samples.tex['omegab'] = '$\Omega_b$'           # Label omegab

    samples.plot_1d('omegab')                      # Simple 1D plot
    
3. Many KDE plotting tools have conventions that don't play well with uniformly distributed parameters, which is a pain if you are trying to plot priors along with your posteriors. anesthetic a sensible mechanism, by defining the contours by the amount of iso-probability mass they contain, but colouring the fill in relation to the probability density of the contour.

To Do
-----
- tests
- CI
- docstrings
- better interfaces for plotting multiple samples
- better interfaces for prior + posterior
- multiple nested sampler input formats (MultiNest, Dynesty, NeuralNest)
- Read multiple MCMC chains
- Automatic coloring
- Legends
- resurrect rhinestone
