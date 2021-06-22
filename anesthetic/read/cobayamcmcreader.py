"""Tools for reading from getdist chains files."""
from anesthetic.read.getdistreader import GetDistReader
try:
    from getdist import loadMCSamples
except ImportError as err:
    raise ImportError("Need GetDist in order to read Cobaya MCMC output.\n"
                      "%s" % err)


class CobayaMCMCReader(GetDistReader):
    """Read Cobaya yaml files."""

    def paramnames(self):
        r"""Read header of <root>.1.txt to infer the paramnames.

        This is the data file of the first chain. It should have as many
        columns as there are parameters (sampled and derived) plus an
        additional two corresponding to the weights (first column) and the
        logposterior (second column). The first line should start with a # and
        should list the parameter names corresponding to the columns. This
        will be used as label in the pandas array. Uses GetDist's
        `loadMCSamples` to infer the tex dictionary.
        """
        try:
            with open(self.root + ".1.txt") as f:
                header = f.readline()[1:]
                paramnames = header.split()[2:]
                s = loadMCSamples(file_root=self.root)
                tex = {i.name: '$' + i.label + '$' for i in s.paramNames.names}
                return paramnames, tex
        except IOError:
            return super().paramnames()

    @property
    def yaml_file(self):
        """Cobaya parameter file."""
        return self.root + '.updated.yaml'
