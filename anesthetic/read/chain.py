"""Read MCMCSamples or NestedSamples from any chains."""
from anesthetic.read.polychord import read_polychord
from anesthetic.read.getdist import read_getdist
from anesthetic.read.cobaya import read_cobaya
from anesthetic.read.multinest import read_multinest


def read_chains(root, *args, **kwargs):
    """Auto-detect chain type and read from file.

    Parameters
    ----------
    root: str
        root name for reading files

    *args, **kwargs:
        Passed onto NestedSamples or MCMCSamples. Check their docstrings for
        more information.

    Returns
    -------
    NestedSamples or MCMCSamples depending on auto-detection

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
    for read in [read_polychord, read_multinest, read_cobaya, read_getdist]:
        try:
            samples = read(root, *args, **kwargs)
            samples.root = root
            return samples
        except (FileNotFoundError, IOError) as e:
            errors.append(str(read) + ": " + str(e))

    errors = ["Could not find any compatible chains:"] + errors
    raise FileNotFoundError('\n'.join(errors))
