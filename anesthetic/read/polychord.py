"""Tools for reading from polychord chains files."""
import numpy
from anesthetic.read.getdist import GetDistReader


class PolyChordReader(GetDistReader):
    """Read polychord files."""

    def samples(self):
        """Read ``<root>_dead-birth.txt`` in polychord format."""
        data = numpy.loadtxt(self.birth_file)
        samples, logL, logL_birth = numpy.split(data, [-2, -1], axis=1)
        return samples, logL.flatten(), logL_birth.flatten()

    @property
    def birth_file(self):
        """File containing dead and birth contours."""
        return self.root + '_dead-birth.txt'
