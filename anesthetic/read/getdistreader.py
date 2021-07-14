"""Tools for reading from getdist chains files."""
import numpy as np
import glob
from anesthetic.read.chainreader import ChainReader


class GetDistReader(ChainReader):
    """Read getdist files."""

    def paramnames(self):
        r"""Read <root>.paramnames in getdist format.

        This file should contain one or two columns. The first column indicates
        a reference name for the sample, used as labels in the pandas array.
        The second optional column should include the equivalent axis label,
        possibly in tex, with the understanding that it will be surrounded by
        dollar signs, for example

        <root.paramnames>

        a1     a_1
        a2     a_2
        omega  \omega
        """
        try:
            with open(self.paramnames_file, 'r') as f:
                paramnames = []
                tex = {}
                for line in f:
                    line = line.strip().split()
                    paramname = line[0].replace('*', '')
                    paramnames.append(paramname)
                    if len(line) > 1:
                        tex[paramname] = '$' + ' '.join(line[1:]) + '$'
                return paramnames, tex
        except IOError:
            return super().paramnames()

    def limits(self):
        """Read <root>.ranges in getdist format."""
        try:
            with open(self.ranges_file, 'r') as f:
                limits = {}
                for line in f:
                    line = line.strip().split()
                    paramname = line[0]
                    try:
                        xmin = float(line[1])
                    except ValueError:
                        xmin = None
                    try:
                        xmax = float(line[2])
                    except ValueError:
                        xmax = None
                    limits[paramname] = (xmin, xmax)
                return limits
        except IOError:
            return super().limits()

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
        weights, minuslogL, samples = np.split(data, [1, 2], axis=1)
        return weights.flatten(), -minuslogL.flatten(), samples

    @property
    def paramnames_file(self):
        """File containing parameter names."""
        return self.root + '.paramnames'

    @property
    def ranges_file(self):
        """File containing parameter names."""
        return self.root + '.ranges'

    @property
    def chains_files(self):
        """File containing parameter names."""
        files = glob.glob(self.root + '_[0-9].txt')
        if not files:
            files = glob.glob(self.root + '.[0-9].txt')
        if not files:
            files = [self.root + '.txt']
        return files
