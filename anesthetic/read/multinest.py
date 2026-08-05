"""Read NestedSamples from MultiNest chains."""
import os
import numpy as np
from anesthetic.read._utils import _norm_columns
from anesthetic.read.getdist import read_getdist_paramnames
from anesthetic.samples import NestedSamples


def read_multinest(root, *args, columns=None, **kwargs):
    """Read MultiNest chain files.

    Parameters
    ----------
    root : str
        Root name for reading files in MultiNest format, i.e. the files
        ``<root>dead-birth.txt`` and ``<root>phys_live-birth.txt`` in the new
        format, and ``<root>ev.dat`` and ``<root>phys_live.points`` in the old
        format.

    columns : list[str], list[int], or slice, optional
        Optionally select which parameter columns to load from the chain files.
        This is useful when you do not want to load a large number of nuisance
        parameters into memory. Integer positions and slices index parameter
        fields only, not sampler bookkeeping fields such as ``logL``.

    *args, **kwargs
        Passed on to ``NestedSamples``. Check its docstring for more
        information.

    Returns
    -------
    :class:`anesthetic.samples.NestedSamples`

    """
    dead_birth_file = root + 'dead-birth.txt'
    dead_file = root + 'ev.dat'
    if not os.path.exists(dead_birth_file) and not os.path.exists(dead_file):
        raise FileNotFoundError(f"{dead_birth_file} or {dead_file} not found.")

    parameters, labels = read_getdist_paramnames(root)
    labels = kwargs.pop('labels', labels)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    indices, columns = _norm_columns(columns, parameters)
    indices = list(range(len(parameters))) if indices is None else indices

    if os.path.exists(dead_birth_file):
        usecols = indices + [-4, -3]
        _data = np.loadtxt(dead_birth_file, usecols=usecols, ndmin=2)
        data, logL, logL_birth = np.split(_data, [-2, -1], axis=1)
        logL = logL[:, 0]
        logL_birth = logL_birth[:, 0]

        phys_live_birth_file = root + 'phys_live-birth.txt'
        usecols = usecols[:-2] + [-3, -2]
        _data = np.loadtxt(phys_live_birth_file, usecols=usecols, ndmin=2)
        live_data, live_logL, live_birth = np.split(_data, [-2, -1],
                                                    axis=1)
        live_logL = live_logL[:, 0]
        live_birth = live_birth[:, 0]
        i = np.argsort(live_logL)
        data = np.concatenate((data, live_data[i]), axis=0)
        logL = np.concatenate((logL, live_logL[i]))
        logL_birth = np.concatenate((logL_birth, live_birth[i]))

    else:
        usecols = indices + [-3]
        _data = np.loadtxt(dead_file, usecols=usecols, ndmin=2)
        data, logL = np.split(_data, [-1], axis=1)
        logL = logL[:, 0]

        usecols = usecols[:-1] + [-2]
        _data = np.loadtxt(root + 'phys_live.points', usecols=usecols, ndmin=2)
        live_data, live_logL = np.split(_data, [-1], axis=1)
        live_logL = live_logL[:, 0]
        i = np.argsort(live_logL)
        logL_birth = len(live_logL)
        data = np.concatenate((data, live_data[i]), axis=0)
        logL = np.concatenate((logL, live_logL[i]))

    return NestedSamples(data=data, columns=columns,
                         logL=logL, logL_birth=logL_birth,
                         labels=labels, *args, **kwargs)
