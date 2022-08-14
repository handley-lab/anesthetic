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
        root name

    Returns
    -------
    NestedSamples or MCMCSamples depending on auto-detection

    """
    errors = []
    for read in [read_polychord, read_getdist, read_cobaya, read_multinest]:
        try:
            return read(root, *args, **kwargs)
        except FileNotFoundError as e:
            errors.append(str(read) + ": " + str(e))

    errors = ["Could not find any compatible chains:"] + errors
    raise FileNotFoundError('\n'.join(errors))
