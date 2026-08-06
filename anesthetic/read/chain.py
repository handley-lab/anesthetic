"""Read MCMCSamples or NestedSamples from any chains."""
from anesthetic.read.polychord import read_polychord
from anesthetic.read.getdist import read_getdist, read_getdist_paramnames
from anesthetic.read.cobaya import read_cobaya, read_cobaya_paramnames
from anesthetic.read.multinest import read_multinest
from anesthetic.read.ultranest import read_ultranest, read_ultranest_paramnames
from anesthetic.read.nestedfit import read_nestedfit, read_nestedfit_paramnames
from anesthetic.read.csv import read_csv


def read_paramnames(root):
    """Read parameter names and labels without loading full chains.

    Parameters
    ----------
    root : str, pathlib.Path
        Root name for reading chain metadata.

    Returns
    -------
    parameters : list[str] or list[int]
        Parameter names in file order, excluding sampler bookkeeping fields.
    labels : dict
        Mapping from parameter names to axis labels.

    """
    root = str(root)
    errors = []
    readers = [read_cobaya_paramnames, read_getdist_paramnames,
               read_nestedfit_paramnames, read_ultranest_paramnames]
    for read in readers:
        try:
            return read(root)
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

    When installed, `GetDist <https://getdist.readthedocs.io/en/latest/>`__
    is used to read parameter labels from Cobaya's YAML metadata.

    Parameters
    ----------
    root : str, pathlib.Path
        Root name for reading chain files.

    columns : list[str], list[int], or slice, optional
        Optionally select which parameter columns to load from the chain files.
        This is useful when you do not want to load a large number of nuisance
        parameters into memory. Integer positions and slices index parameter
        fields only, not sampler bookkeeping fields such as ``logL``.

    renames : dict, optional
        Mapping from parameter names to new names.

    burn_in : int, float or array-like, optional
        For Cobaya and GetDist MCMC chains:
        Number or fraction of stored rows to remove from each chain before
        loading samples into memory. Uses the same semantics as
        :meth:`anesthetic.samples.MCMCSamples.remove_burn_in`.

    thin : int, optional
        For Cobaya and GetDist MCMC chains:
        Keep every ``thin``-th sample in the expanded MCMC chain represented
        by the frequency weights.

    compress_repeats : bool, default=False
        For Cobaya and GetDist MCMC chains:
        Oversampling nuisance parameters can leave the selected parameters of
        interest unchanged across consecutive samples. Merge these repeated
        rows by summing their weights. Compression happens separately for each
        chain, after burn-in removal and thinning. If ``False``, likelihood
        bookkeeping fields such as ``logL`` are returned in addition to the
        selected columns. If ``True``, only the selected columns and ``chain``
        are returned. Weights are always retained.

    *args, **kwargs
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
