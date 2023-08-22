"""Read MCMCSamples or NestedSamples from any chains."""
from anesthetic.read.polychord import read_polychord
from anesthetic.read.getdist import read_getdist
from anesthetic.read.cobaya import read_cobaya
from anesthetic.read.multinest import read_multinest
from anesthetic.read.ultranest import read_ultranest
from anesthetic.read.nestedfit import read_nestedfit


def read_chains(root, *args, **kwargs):
    """Auto-detect chain type and read from file.

    anesthetic supports chains from:

        * `PolyChord <https://github.com/PolyChord/PolyChordLite>`_,
        * `MultiNest <https://github.com/farhanferoz/MultiNest>`_,
        * `UltraNest <https://github.com/JohannesBuchner/UltraNest>`_,
        * `Nested_fit <https://github.com/martinit18/Nested_Fit>`_,
        * `CosmoMC <https://github.com/cmbant/CosmoMC>`_,
        * `Cobaya <https://github.com/CobayaSampler/cobaya>`_,
        * or anything `GetDist <https://github.com/cmbant/getdist>`_
          compatible.

    Note that in order to optimally read chains from Cobaya you need to have
    `GetDist <https://getdist.readthedocs.io/en/latest/>`__ installed.

    Parameters
    ----------
    root : str
        root name for reading files

    *args, **kwargs:
        Passed onto ``NestedSamples`` or ``MCMCSamples``. Check their
        docstrings for more information.

    Returns
    -------
    :class:`anesthetic.samples.NestedSamples` or
    :class:`anesthetic.samples.MCMCSamples` depending on auto-detection

    """
    if 'burn_in' in kwargs:
        raise KeyError(
            "This is anesthetic 1.0 syntax. The `burn_in` keyword is no "
            "longer supported in `read_chains`. You need to update, e.g.\n"
            "read_chains(root, burn_in=0.5)         # anesthetic 1.0\n"
            "read_chains(root).remove_burn_in(0.5)  # anesthetic 2.0\n"
            "See also https://anesthetic.readthedocs.io/en/latest/"
            "anesthetic.html#anesthetic.samples.MCMCSamples.remove_burn_in"
        )
    errors = []
    readers = [
        read_polychord, read_multinest, read_cobaya,
        read_ultranest, read_nestedfit, read_getdist
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
