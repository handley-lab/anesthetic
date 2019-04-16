"""Tools for reading from polychord chains files."""
import numpy
from anesthetic.read.getdist import GetDistReader

class PolyChordReader(GetDistReader):
    """Read polychord files."""

    def samples(self):
        """Read ``<root>_dead-birth.txt`` in polychord format."""
        birth_file = self.root + '_dead-birth.txt'
        data = numpy.loadtxt(birth_file)
        samples, logL, logL_birth = numpy.split(data, [-2, -1], axis=1)
        return samples, logL, logL_birth
