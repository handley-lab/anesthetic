"""Read MCMCSamples or NestedSamples from any chains."""
from anesthetic.read.polychord import read_polychord
from anesthetic.read.getdist import (
    read_getdist, read_paramnames as read_getdist_paramnames
)
from anesthetic.read.cobaya import (
    read_cobaya, read_paramnames as read_cobaya_paramnames
)
from anesthetic.read.multinest import read_multinest
from anesthetic.read.ultranest import read_ultranest
from anesthetic.read.nestedfit import read_nestedfit
from anesthetic.read.csv import read_csv


def read_parameters(root):
    """Read parameter names without loading samples into memory.

    Parameter discovery is supported for Cobaya chain headers and GetDist
    ``.paramnames`` files. The latter also covers formats such as PolyChord
    and MultiNest that use GetDist parameter metadata.

    Parameters
    ----------
    root : str, pathlib.Path
        Root name for reading chain metadata.

    Returns
    -------
    list of str
        Parameter names in file order, excluding sampler bookkeeping columns.

    """
    root = str(root)
    errors = []
    readers = [read_cobaya_paramnames, read_getdist_paramnames]
    for read in readers:
        try:
            parameters, _ = read(root)
            if parameters is not None:
                return parameters
        except (FileNotFoundError, IOError) as error:
            errors.append(str(read) + ": " + str(error))

    errors = ["Could not find any compatible parameter metadata:"] + errors
    raise FileNotFoundError('\n'.join(errors))


def read_chains(root, *args, **kwargs):
    """Auto-detect chain type and read from file.

    anesthetic supports chains from:

        * `PolyChord <https://github.com/PolyChord/PolyChordLite>`_,
        * `MultiNest <https://github.com/farhanferoz/MultiNest>`_,
        * `UltraNest <https://github.com/JohannesBuchner/UltraNest>`_,
        * `Nested_fit <https://github.com/martinit18/Nested_Fit>`_,
        * `CosmoMC <https://github.com/cmbant/CosmoMC>`_,
        * `Cobaya <https://github.com/CobayaSampler/cobaya>`_,
        * anything `GetDist <https://github.com/cmbant/getdist>`_ compatible,
        * files produced using ``DataFrame.to_csv()`` from anesthetic.

    Note that in order to optimally read chains from Cobaya you need to have
    `GetDist <https://getdist.readthedocs.io/en/latest/>`__ installed.

    Parameters
    ----------
    root : str, pathlib.Path
        root name for reading files

    columns : list[str], list[int], or slice, optional
        For Cobaya chains, optionally select which parameter columns to load
        from the chain files. This is useful when you do not want to load a
        large number of nuisance parameters into memory. Integer positions and
        slices index the parameter names returned by :func:`read_parameters`.
        Weights and the ``chi2``, ``logP``, ``logL``, and ``chain`` columns are
        always included.

    *args, **kwargs:
        Passed on to ``NestedSamples`` or ``MCMCSamples``. Check their
        docstrings for more information.

    Returns
    -------
    :class:`anesthetic.samples.NestedSamples` or
    :class:`anesthetic.samples.MCMCSamples` depending on auto-detection

    """
    root = str(root)
    errors = []
    readers = [
        read_polychord, read_multinest, read_cobaya, read_ultranest,
        read_nestedfit, read_getdist, read_csv
    ]
    for read in readers:
        try:
            samples = read(root, *args, **kwargs)
            samples.root = root
            return samples
        except (FileNotFoundError, IOError) as e:
            errors.append(str(read) + ": " + str(e))

    errors = ["Could not find any compatible chains:"] + errors
    raise FileNotFoundError('\n'.join(errors))
