"""Read NestedSamples from MultiNest chains."""
import os
import numpy as np
from anesthetic.read.getdist import read_paramnames
from anesthetic.samples import NestedSamples


def read_multinest(root, *args, **kwargs):
    """Read MultiNest chain files.

    Parameters
    ----------
    root : str
        root name for reading files in MultiNest format, i.e. the files
        ``<root>ev.dat`` and ``<root>phys_live.points`` in the old format, and
        ``<root>dead-birth.txt`` and ``<root>phys_live-birth.txt`` in the new
        format.

    """
    try:
        data = np.loadtxt(root + 'dead-birth.txt')
        samples, logL, logL_birth, _ = np.split(data, [-4, -3, -2], axis=1)
        logL = np.squeeze(logL)
        logL_birth = np.squeeze(logL_birth)

        data = np.loadtxt(root + 'phys_live-birth.txt')
        (live_samples, live_logL,
         live_logL_birth, _) = np.split(data, [-3, -2, -1], axis=1)
        live_logL = np.squeeze(live_logL)
        live_logL_birth = np.squeeze(live_logL_birth)
        i = np.argsort(live_logL)
        samples = np.concatenate((samples, live_samples[i]), axis=0)
        logL = np.concatenate((logL, live_logL[i]))
        logL_birth = np.concatenate((logL_birth, live_logL_birth[i]))

    except (FileNotFoundError, IOError):
        data = np.loadtxt(root + 'ev.dat')
        samples, logL, _ = np.split(data, [-3, -2], axis=1)
        logL = np.squeeze(logL)

        data = np.loadtxt(root + 'phys_live.points')
        live_samples, live_logL, _ = np.split(data, [-2, -1], axis=1)
        live_logL = np.squeeze(live_logL)
        i = np.argsort(live_logL)
        logL_birth = len(live_logL)
        samples = np.concatenate((samples, live_samples[i]), axis=0)
        logL = np.concatenate((logL, live_logL[i]))

    kwargs['label'] = kwargs.get('label', os.path.basename(root))
    columns, labels = read_paramnames(root)
    columns = kwargs.pop('columns', columns)
    labels = kwargs.pop('labels', labels)
    data = samples

    return NestedSamples(data=data, logL=logL, logL_birth=logL_birth,
                         columns=columns, labels=labels, *args, **kwargs)
