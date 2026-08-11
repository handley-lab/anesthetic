"""Read NestedSamples from PolyChord chains."""
import os
import numpy as np
from anesthetic.read._utils import _norm_columns
from anesthetic.read.getdist import read_getdist_paramnames
from anesthetic.samples import NestedSamples


def read_polychord(root, *args, columns=None, renames=None, **kwargs):
    """Read PolyChord chain files.

    Parameters
    ----------
    root : str
        Root name for reading files in PolyChord format, i.e. the files
        ``<root>_dead-birth.txt`` and ``<root>_phys_live-birth.txt``.

    columns : list[str], list[int], or slice, optional
        Optionally select which parameter columns to load from the chain files.
        This is useful when you do not want to load a large number of nuisance
        parameters into memory. Integer positions and slices index parameter
        fields only, not sampler bookkeeping fields such as ``logL``.

    renames : dict, optional
        Mapping from parameter names to new names.

    *args, **kwargs
        Passed on to ``NestedSamples``. Check its docstring for more
        information.

    Returns
    -------
    :class:`anesthetic.samples.NestedSamples`

    """
    dead_birth_file = root + '_dead-birth.txt'
    if not os.path.exists(dead_birth_file):
        raise FileNotFoundError(f"{dead_birth_file} not found.")

    parameters, labels = read_getdist_paramnames(root)
    labels = kwargs.pop('labels', labels)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    indices, columns, renames = _norm_columns(columns, parameters, renames)
    usecols = None if indices is None else indices + [-2, -1]

    data = np.loadtxt(dead_birth_file, usecols=usecols, ndmin=2)
    try:
        phys_live_birth_file = root + '_phys_live-birth.txt'
        _data = np.loadtxt(phys_live_birth_file, usecols=usecols, ndmin=2)
        data = np.concatenate([data, _data]) if _data.size else data
        data = np.unique(data, axis=0)
        i = np.argsort(data[:, -2])
        data = data[i, :]
    except IOError:
        pass
    data, logL, logL_birth = np.split(data, [-2, -1], axis=1)
    columns = [renames.get(column, column) for column in columns]

    return NestedSamples(data=data, columns=columns,
                         logL=logL, logL_birth=logL_birth,
                         labels=labels, *args, **kwargs)
