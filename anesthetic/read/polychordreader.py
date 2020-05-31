"""Tools for reading from polychord chains files."""
import numpy as np
from anesthetic.read.getdistreader import GetDistReader


class PolyChordReader(GetDistReader):
    """Read polychord files."""

    def samples(self):
        """Read ``<root>_dead-birth.txt`` in polychord format."""
        data = np.loadtxt(self.birth_file)
        try:
            _data = np.loadtxt(self.phys_live_birth_file)
            data = np.concatenate([data, _data])
            data = np.unique(data, axis=0)
            i = np.argsort(data[:, -2])
            data = data[i, :]
        except (OSError, IOError):
            pass
        samples, logL, logL_birth = np.split(data, [-2, -1], axis=1)
        return samples, logL.flatten(), logL_birth.flatten()

    @property
    def birth_file(self):
        """File containing dead and birth contours."""
        return self.root + '_dead-birth.txt'

    @property
    def phys_live_birth_file(self):
        """File containing physical live points."""
        return self.root + '_phys_live-birth.txt'
