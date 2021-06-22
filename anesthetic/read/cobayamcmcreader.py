"""Tools for reading from getdist chains files."""
from warnings import warn
from anesthetic.read.getdistreader import GetDistReader
try:
    from getdist import loadMCSamples
except ImportError as imperr:
    pass


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
                tex = {p.name: '$' + p.label + '$' for p in s.paramNames.names}
                return paramnames, tex
        except IOError:
            return super().paramnames()

    def limits(self):
        """Infer param limits from <root>.yaml in cobaya format."""
        s = loadMCSamples(file_root=self.root)
        return {p.name: (s.ranges.getLower(p.name), s.ranges.getUpper(p.name))
                for p in s.paramNames.names
                if s.ranges.getLower(p.name) is not None
                or s.ranges.getUpper(p.name) is not None}

    @property
    def yaml_file(self):
        """Cobaya parameter file."""
        return self.root + '.updated.yaml'
