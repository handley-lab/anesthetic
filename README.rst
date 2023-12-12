===========================================
anesthetic: nested sampling post-processing
===========================================
:Authors: Will Handley and Lukas Hergt
:Version: 2.7.0
:Homepage: https://github.com/handley-lab/anesthetic
:Documentation: http://anesthetic.readthedocs.io/

.. image:: https://github.com/handley-lab/anesthetic/workflows/CI/badge.svg?branch=master
   :target: https://github.com/handley-lab/anesthetic/actions?query=workflow%3ACI+branch%3Amaster
   :alt: Build Status
.. image:: https://codecov.io/gh/handley-lab/anesthetic/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/handley-lab/anesthetic
   :alt: Test Coverage Status
.. image:: https://readthedocs.org/projects/anesthetic/badge/?version=latest
   :target: https://anesthetic.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://badge.fury.io/py/anesthetic.svg
   :target: https://badge.fury.io/py/anesthetic
   :alt: PyPi location
.. image:: https://zenodo.org/badge/175663535.svg
   :target: https://zenodo.org/badge/latestdoi/175663535
   :alt: Permanent DOI for this release
.. image:: http://joss.theoj.org/papers/8c51bffda75d122cf4a8b991e18d3e45/status.svg
   :target: http://joss.theoj.org/papers/8c51bffda75d122cf4a8b991e18d3e45
   :alt: Review Status
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/handley-lab/anesthetic/blob/master/LICENSE
   :alt: License information





``anesthetic`` brings together tools for processing nested sampling chains by leveraging standard scientific python libraries.

TLDR: See the `quickstart <https://anesthetic.readthedocs.io/en/latest/quickstart.html>`__.

Current functionality includes:

- Computation of Bayesian evidences, Kullback-Liebler divergences and Bayesian model dimensionalities.
- Marginalised 1d and 2d plots.
- Dynamic replaying of nested sampling.

This tool was designed primarily for use with nested sampling outputs, although it can be used for normal MCMC chains.

For an interactive view of a nested sampling run, you can use the ``anesthetic`` script.

.. code:: bash

   $ anesthetic <ns file root>

.. image:: https://github.com/handley-lab/anesthetic/raw/master/images/anim_1.gif

Features
--------

- Both samples and plotting axes are stored as a ``pandas.DataFrame``, with parameter names as indices, which makes for easy access and modification.
- Sensible color scheme for plotting nearly flat distributions.
- For easy extension/modification, uses the standard python libraries:
  `numpy <https://www.numpy.org/>`__,
  `scipy <https://www.scipy.org/>`__,
  `matplotlib <https://matplotlib.org/>`__
  and `pandas <https://pandas.pydata.org/>`__.

Installation
------------

``anesthetic`` can be installed via pip

.. code:: bash

    pip install anesthetic

or from the repository

.. code:: bash

    git clone https://github.com/handley-lab/anesthetic
    cd anesthetic
    python -m pip install .

You can check that things are working by running the test suite:

.. code:: bash

    export MPLBACKEND=Agg     # only necessary for OSX users
    python -m pip install ".[test]"
    python -m pytest
    flake8 anesthetic tests
    pydocstyle --convention=numpy anesthetic


Dependencies
~~~~~~~~~~~~

Basic requirements:

- Python 3.9+
- `matplotlib <https://pypi.org/project/matplotlib/>`__
- `numpy <https://pypi.org/project/numpy/>`__
- `scipy <https://pypi.org/project/scipy/>`__
- `pandas <https://pypi.org/project/pandas/>`__
- `fastKDE <https://pypi.org/project/fastkde/>`__

Documentation:

- `sphinx <https://pypi.org/project/Sphinx/>`__
- `numpydoc <https://pypi.org/project/numpydoc/>`__

Tests:

- `pytest <https://pypi.org/project/pytest/>`__

Documentation
-------------

Full Documentation is hosted at `ReadTheDocs <http://anesthetic.readthedocs.io/>`__.  To build your own local copy of the documentation you'll need to install `sphinx <https://pypi.org/project/Sphinx/>`__. You can then run:

.. code:: bash

    python -m pip install ".[all,docs]"
    cd docs
    make html

and view the documentation by opening ``docs/build/html/index.html`` in a browser. To regenerate the automatic RST files run:

.. code:: bash

    sphinx-apidoc -fM -t docs/templates/ -o docs/source/ anesthetic/

Citation
--------

If you use ``anesthetic`` to generate plots for a publication, please cite
as: ::

   Handley, (2019). anesthetic: nested sampling visualisation. Journal of Open
   Source Software, 4(37), 1414, https://doi.org/10.21105/joss.01414

or using the BibTeX:

.. code:: bibtex

   @article{anesthetic,
       doi = {10.21105/joss.01414},
       url = {http://dx.doi.org/10.21105/joss.01414},
       year  = {2019},
       month = {Jun},
       publisher = {The Open Journal},
       volume = {4},
       number = {37},
       pages = {1414},
       author = {Will Handley},
       title = {anesthetic: nested sampling visualisation},
       journal = {The Journal of Open Source Software}
   }


Contributing
------------
There are many ways you can contribute via the `GitHub repository <https://github.com/handley-lab/anesthetic>`__.

- You can `open an issue <https://github.com/handley-lab/anesthetic/issues>`__ to report bugs or to propose new features.
- Pull requests are very welcome. Note that if you are going to propose major changes, be sure to open an issue for discussion first, to make sure that your PR will be accepted before you spend effort coding it.


Questions/Comments
------------------
Another posterior plotting tool?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This is my posterior plotter. There are many like it, but this one is mine.

There are several excellent tools for plotting marginalised posteriors:

- `getdist <http://getdist.readthedocs.io/en/latest/intro.html>`__
- `corner <https://corner.readthedocs.io>`__
- `pygtc <https://pygtc.readthedocs.io>`__
- `dynesty <https://dynesty.readthedocs.io>`__
- `MontePython <http://baudren.github.io/montepython.html>`__

Why create another one? In general, any dedicated user of software will find that there is some functionality that in their use case is lacking, and the designs of previous codes make such extensions challenging. In my case this was:

1. For large numbers of samples, kernel density estimation is slow, or inaccurate (particularly for samples generated from nested sampling). There are kernel density estimators, such as `fastKDE <https://pypi.org/project/fastkde/>`__, which ameliorate many of these difficulties.

2. Existing tools can make it difficult to define new parameters. For example, the default cosmomc chain defines ``omegabh2``, but not ``omegab``. The transformation is easy, since ``omegab = omegabh2/ (H0/100)**2``, but implementing this transformation in existing packages is not so trivial. ``anesthetic`` solves this issue by storing the samples as a pandas array, for which the relevant code for defining the above new parameter would be

.. code:: python

    from anesthetic import read_chains
    samples = read_chains(file_root)               # Load the samples
    label = 'omegab'
    tex = '$\Omega_b$'
    h = (samples.H0/100)
    samples[(label, tex)] = samples.omegabh2/h**2  # Define omegab
    samples.plot_1d('omegab')                      # Simple 1D plot

3. Many KDE plotting tools have conventions that don't play well with uniformly distributed parameters, which presents a problem if you are trying to plot priors along with your posteriors. ``anesthetic`` has a sensible mechanism, by defining the contours by the amount of iso-probability mass they contain, but colouring the fill in relation to the probability density of the contour.

What's in a name?
~~~~~~~~~~~~~~~~~

There is a convention for naming nested sampling packages with words that have nest in them (`nestle and dynesty <https://dynesty.readthedocs.io/en/latest/>`__, `nestorflow <https://bitbucket.org/tomcharnock/nestorflow>`__). Doing a UNIX grep:

.. code:: bash

    grep nest /usr/share/dict/words

yields a lot of superlatives (e.g. greenest), but a few other cool names for future projects:

- amnesty
- defenestrate
- dishonestly
- inestimable
- minestrone
- rhinestone

I chose ``anesthetic`` because I liked the soft 'th', and in spite of the US spelling.
