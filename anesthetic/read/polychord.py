"""Read NestedSamples from PolyChord chains."""
import os
import numpy as np
from anesthetic.read.getdist import read_paramnames
from anesthetic.samples import NestedSamples


def read_polychord(root, *args, **kwargs):
    """Read PolyChord chain files.

    Parameters
    ----------
    root : str
        root name for reading files in PolyChord format, i.e. the files
        ``<root>_dead-birth.txt`` and ``<root>_phys_live-birth.txt``.

    """
    birth_file = root + '_dead-birth.txt'
    birth_file
    data = np.loadtxt(birth_file)
    try:
        phys_live_birth_file = root + '_phys_live-birth.txt'
        _data = np.loadtxt(phys_live_birth_file)
        _data = np.atleast_2d(_data)
        data = np.concatenate([data, _data]) if _data.size else data
        data = np.unique(data, axis=0)
        i = np.argsort(data[:, -2])
        data = data[i, :]
    except IOError:
        pass
    data, logL, logL_birth = np.split(data, [-2, -1], axis=1)
    columns, labels = read_paramnames(root)

    columns = kwargs.pop('columns', columns)
    labels = kwargs.pop('labels', labels)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    return NestedSamples(data=data, columns=columns,
                         logL=logL, logL_birth=logL_birth,
                         labels=labels, *args, **kwargs)
