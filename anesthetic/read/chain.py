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

    burn_in: float
        if 0 < burn_in < 1:
            discard the first burn_in fraction of samples
        elif 1 < burn_in:
            only keep samples [burn_in:]
        Only works if `root` provided and if chains are GetDist or Cobaya
        compatible.
        default: False

    *args, **kwargs:
        Passed onto NestedSamples or MCMCSamples. Check their docstrings for
        more information.

    Returns
    -------
    NestedSamples or MCMCSamples depending on auto-detection

    """
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
