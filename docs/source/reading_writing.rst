*******************
Reading and writing
*******************


.. _reading chains:

Reading chain files from PolyChord, MultiNest, UltraNest, CosmoMC, or Cobaya
============================================================================

If you have finished nested sampling or MCMC runs from one of:

* `PolyChord <https://polychord.io>`_: https://github.com/PolyChord/PolyChordLite
* MultiNest: https://github.com/farhanferoz/MultiNest
* `UltraNest <https://johannesbuchner.github.io/UltraNest/index.html>`_: https://github.com/JohannesBuchner/UltraNest
* `CosmoMC <https://cosmologist.info/cosmomc/readme.html>`_: https://github.com/cmbant/CosmoMC
* `Cobaya <https://cobaya.readthedocs.io>`_: https://github.com/CobayaSampler/cobaya

then you should be able to read in the chain files directly, by passing the
``root`` to the :func:`anesthetic.read.chain.read_chains` function.

Feel free to use the testing data in ``anesthetic/tests/example_data`` to try
out the examples listed here.

* PolyChord samples, which will be an instance of the
  :class:`anesthetic.samples.NestedSamples` class:

  ::
      
      from anesthetic import read_chains
      samples = read_chains("anesthetic/tests/example_data/pc")

* UltraNest samples, which will be an instance of the
  :class:`anesthetic.samples.NestedSamples` class:

  ::
      
      from anesthetic import read_chains
      samples = read_chains("anesthetic/tests/example_data/un")

* Cobaya samples, which will be an instance of the
  :class:`anesthetic.samples.MCMCSamples` class:

  ::
      
      from anesthetic import read_chains
      samples = read_chains("anesthetic/tests/example_data/cb").remove_burn_in(burn_in=0.1)


.. _passing data:

Passing data as arguments
=========================

You can also pass your own (weighted) data directly to the main sample classes
:class:`anesthetic.samples.Samples`, :class:`anesthetic.samples.MCMCSamples`,
or :class:`anesthetic.samples.NestedSamples`, e.g. here with randomly generated
data:

::

    import numpy as np
    from scipy.stats import multivariate_normal as mvn
    from anesthetic.samples import Samples

    num_samples = 1000                    # number of samples
    num_dim = 2                           # number of parameters/dimensions
    params = ['a', 'b']
    data = np.random.uniform(-5, 5, size=(num_samples, num_dim))
    weights = mvn.pdf(data, mean=[0, 0], cov=np.diag([1, 1]))
    samples = Samples(data, weights=weights, columns=params)


Saving ``NestedSamples`` or ``MCMCSamples``
===========================================

In principle you can use any of pandas options for saving your samples.
However, since our weighted samples here make heavy use of
:class:`pandas.MultiIndex` feature, we recommended using either ``CSV`` files
or ``parquet`` files for reading and writing.

* ``samples.to_csv("filename.csv")``: ``CSV`` files are a useful option when
  you would like to have a human readable file. Check out
  :meth:`pandas.DataFrame.to_csv` for the various options of saving the data
  (e.g. choosing the delimiter etc.).

* ``samples.to_hdf("filename.h5", "samples")``: When reading and writing speed
  is an issue, we recommend using the ``hdf5`` file format, which should be
  faster than ``to_csv`` while still capable of handling the
  :class:`pandas.MultiIndex` format.


Loading ``NestedSamples`` or ``MCMCSamples``
============================================

When loading in previously saved samples from csv, make sure to use the
appropriate class: ``Samples``, ``MCMCSamples``, or ``NestedSamples``.

* ``read_csv``:

  ::
  
      from pandas import read_csv
      from anesthetic import Samples  # or MCMCSamples, or NestedSamples
      samples = Samples(read_csv("filename.csv"))

When loading in previously saved samples from hdf5, make sure to import the
``anesthetic.read_hdf`` function, and not the ``pandas.read_hdf`` version. If
you forget to do this, the samples will be read in as a ``DataFrame``, with a
consequent loss of functionality


* ``read_hdf``:

  ::
  
      from anesthetic import read_hdf
      samples = read_hdf("filename.h5", "samples")


Converting to GetDist
=====================

There are also tools for converting to alternative formats (e.g. `GetDist
<https://getdist.readthedocs.io/en/latest/>`_), in case you have pipelines in
other plotters:

::

    from anesthetic.convert import to_getdist
    getdist_samples = to_getdist(samples)
