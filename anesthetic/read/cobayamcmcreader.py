"""Tools for reading from getdist chains files."""
import glob
import numpy as np
from anesthetic.read.chainreader import ChainReader
try:
    from getdist import loadMCSamples
except ImportError:
    pass


class CobayaMCMCReader(ChainReader):
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

    def samples(self, burn_in=False):
        """Read <root>_1.txt in getdist format."""
        data = np.array([])
        for chains_file in self.chains_files:
            data_i = np.loadtxt(chains_file)
            if burn_in:
                if 0 < burn_in < 1:
                    index = int(len(data_i) * burn_in)
                elif type(burn_in) is int and 1 < burn_in < len(data_i):
                    index = burn_in
                else:
                    raise ValueError("`burn_in` is %s, but should be an "
                                     "integer greater 1 and smaller len(data) "
                                     "or a float between 0 and 1." % burn_in)
                data_i = data_i[index:]
            data = np.concatenate((data, data_i)) if data.size else data_i
        weights, logP, samples = np.split(data, [1, 2], axis=1)
        return weights.flatten(), logP.flatten(), samples

    @property
    def yaml_file(self):
        """Cobaya parameter file."""
        return self.root + '.updated.yaml'

    @property
    def chains_files(self):
        """File containing parameter names."""
        files = glob.glob(self.root + '.[0-99].txt')
        return files
