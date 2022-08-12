from anesthetic.read.polychord import read_polychord
from anesthetic.read.getdist import read_getdist
from anesthetic.read.cobaya import read_cobaya
from anesthetic.read.multinest import read_multinest

def read_chain(root, *args, **kwargs):
    errors = []
    for read in [read_polychord, read_getdist, read_cobaya, read_multinest]:
        try:
            return read(root, *args, **kwargs)
        except FileNotFoundError as e:
            errors.append(str(read) + ": " + str(e))

    errors = ["Could not find any compatible chains:"] + errors
    raise FileNotFoundError('\n'.join(errors))
