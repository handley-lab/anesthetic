*******************
Reading and writing
*******************


.. _reading chains:

Reading chain files from PolyChord, MultiNest, CosmoMC, or Cobaya
=================================================================

If you have finished nested sampling or MCMC runs from PolyChord, MultiNest,
CosmoMC, or Cobaya, then you should be able to read in the chain files
directly, by passing the ``root`` to the
:func:`anesthetic.read.chain.read_chains` function.

Feel free to use the testing data in ``anesthetic/tests/example_data`` to try
out the examples listed here.

* PolyChord samples, which will be an instance of the
  :class:`anesthetic.samples.NestedSamples` class:

  ::
      
      from anesthetic import read_chains
      samples = read_chains("anesthetic/tests/example_data/pc")

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
However, since our weighted samples here make heavy use of pandas multiindex
feature, we recommended using either ``CSV`` files or ``parquet`` files for
reading and writing.

* ``samples.to_csv("filename.csv")``: ``CSV`` files are a useful option when
  you would like to have a human readable file. Check out
  :py:meth:`pandas.DataFrame.to_csv` for the various options of saving the data
  (e.g. choosing the delimiter etc.).

* ``samples.to_parquet("filename.parquet")``: When reading and writing speed is
  an issue, we recommend using the ``parquet`` file format, which should be
  faster than ``to_csv`` while still capable of handling the multiindex format.
  Check out :py:meth:`pandas.DataFrame.to_parquet` for more information.




Loading ``NestedSamples`` or ``MCMCSamples``
============================================

When loading in previously saved samples, make sure to use the appropriate
class: ``Samples``, ``MCMCSamples``, or ``NestedSamples``.

* ``read_csv``:

  ::
  
      from pandas import read_csv
      from anesthetic import Samples  # or MCMCSamples, or NestedSamples
      samples = Samples(read_csv("filename.csv"))

* ``read_parquet``:

  ::
  
      from pandas import read_parquet
      from anesthetic import Samples  # or MCMCSamples, or NestedSamples
      samples = Samples(read_parquet("filename.parquet"))

