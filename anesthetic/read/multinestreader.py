"""Tools for reading from multinest chains files."""
import numpy as np
from anesthetic.read.getdistreader import GetDistReader


class MultiNestReader(GetDistReader):
    """Read multinest files."""

    def samples(self):
        """Read <root>ev.dat and <root>phys_live.points in multinest format."""
        try:
            data = np.loadtxt(self.birth_file)
            samples, logL, logL_birth, _ = np.split(data, [-4, -3, -2], axis=1)
            logL = np.squeeze(logL)
            logL_birth = np.squeeze(logL_birth)

            data = np.loadtxt(self.phys_live_birth_file)
            (live_samples, live_logL,
             live_logL_birth, _) = np.split(data, [-3, -2, -1], axis=1)
            live_logL = np.squeeze(live_logL)
            live_logL_birth = np.squeeze(live_logL_birth)
            i = np.argsort(live_logL)
            samples = np.concatenate((samples, live_samples[i]), axis=0)
            logL = np.concatenate((logL, live_logL[i]))
            logL_birth = np.concatenate((logL_birth, live_logL_birth[i]))
            return samples, logL, logL_birth

        except IOError:
            data = np.loadtxt(self.ev_file)
            samples, logL, _ = np.split(data, [-3, -2], axis=1)
            logL = np.squeeze(logL)

            data = np.loadtxt(self.phys_live_file)
            live_samples, live_logL, _ = np.split(data, [-2, -1], axis=1)
            live_logL = np.squeeze(live_logL)
            i = np.argsort(live_logL)
            nlive = len(live_logL)
            samples = np.concatenate((samples, live_samples[i]), axis=0)
            logL = np.concatenate((logL, live_logL[i]))
            return samples, logL, nlive

    @property
    def ev_file(self):
        """File containing discarded points."""
        return self.root + 'ev.dat'

    @property
    def birth_file(self):
        """File containing discarded points."""
        return self.root + 'dead-birth.txt'

    @property
    def phys_live_file(self):
        """File containing physical live points."""
        return self.root + 'phys_live.points'

    @property
    def phys_live_birth_file(self):
        """File containing physical live points."""
        return self.root + 'phys_live-birth.txt'
