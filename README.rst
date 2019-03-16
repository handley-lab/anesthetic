===========================================
aNESThetic: nested sampling post-processing 
===========================================
:aNESThetic: nested sampling post-processing
:Author: Will Handley
:Version: 0.1.0
:Homepage: https://github.com/williamjameshandley/anesthetic

This project is still in alpha stage. It aims to bring together tools for processing nested sampling chains, leveraging the standard python libraries:

- numpy
- scipy
- pandas

Another triangle plotting tool?
-------------------------------

::
    
    This is my triangle plotter. There are many like it, but this one is mine.

There are several excellent tools for plotting marginalised posteriors:

- `getdist <http://getdist.readthedocs.io/en/latest/intro.html>`__ 
- `corner <https://corner.readthedocs.io/en/latest/>`__
- `MontePython <https://github.com/brinckmann/montepython_public>`__
- `pygtc <https://pygtc.readthedocs.io/en/latest/>`__

Why create another one? In general, any user of software will find that there is some functionality that in their opinion is severely lacking. In my case this was primarily:

First, for large numbers of samples, kernel density estimation is slow, or inaccurate. There are now better state-of-the-art kernel density estimators, such as `fastKDE <https://pypi.org/project/fastkde/>`__, which ameliorate many of these difficulties.

Second, existing tools can make it painfully difficult to define new parameters. Take for example the default cosmomc chain, which defines ``omegabh2``, but not ``omegab``. The transformation is easy, since ``omegab = omegabh2/ (H0/100)**2``, but writing this simple line in code is not easy. This tool solves this issue by storing the samples as a pandas array, for which the relevant code for defining the above new parameter would be

.. code:: python

    from anesthetic.kde import load_nested_samples

    samples = load_nested_samples(file_root)       # Load the samples

    h = samples['H0']/100                          # Define h
    samples['omegab'] = samples.omegabh2/h**2      # Define omegab
    samples.tex['omegab'] = '\Omega_b'             # Label omegab

    samples.plot('omegab')                         # Simple 1D plot

This triangle plotting tool was designed primarily for use with nested sampling, although it can be used for normal MCMC chains.

To Do
-----
- tests
- CI
- docstrings
