"""Read NestedSamples from MultiNest chains."""
import os
import numpy as np
from anesthetic.read._utils import _norm_columns
from anesthetic.read.getdist import read_getdist_paramnames
from anesthetic.samples import NestedSamples


def read_multinest(root, *args, columns=None, renames=None, **kwargs):
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

    renames : dict, optional
        Mapping from parameter names to new names (i.e. column handles).
        Labels are not carried over to renamed parameters, so provide them
        separately via a ``labels`` dict with the new parameter names as keys.

    *args, **kwargs
        Passed on to ``NestedSamples``. Check its docstring for more
        information.

    Returns
    -------
    :class:`anesthetic.samples.NestedSamples`

    """
    dead_birth_file = root + 'dead-birth.txt'
    phys_live_birth_file = root + 'phys_live-birth.txt'
    dead_file = root + 'ev.dat'
    phys_live_file = root + 'phys_live.points'
    if not os.path.exists(dead_birth_file) and not os.path.exists(dead_file):
        raise FileNotFoundError(f"{dead_birth_file} or {dead_file} not found.")

    parameters, labels = read_getdist_paramnames(root)
    indices, columns, renames = _norm_columns(columns, parameters, renames)
    indices = list(range(len(parameters))) if indices is None else indices

    if os.path.exists(dead_birth_file):
        usecols = indices + [-4, -3]
        data = np.loadtxt(dead_birth_file, usecols=usecols, ndmin=2)

        usecols = usecols[:-2] + [-3, -2]
        _data = np.loadtxt(phys_live_birth_file, usecols=usecols, ndmin=2)
        i = np.argsort(_data[:, -2])
        data = np.concatenate([data, _data[i]])
        data, logL, logL_birth = np.split(data, [-2, -1], axis=1)

    else:
        usecols = indices + [-3]
        data = np.loadtxt(dead_file, usecols=usecols, ndmin=2)

        usecols = usecols[:-1] + [-2]
        _data = np.loadtxt(phys_live_file, usecols=usecols, ndmin=2)
        i = np.argsort(_data[:, -1])
        data = np.concatenate([data, _data[i]])
        data, logL = np.split(data, [-1], axis=1)
        logL_birth = len(_data)

    columns = [renames.get(column, column) for column in columns]
    labels = kwargs.pop('labels', labels)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    samples = NestedSamples(data=data, columns=columns,
                            logL=logL, logL_birth=logL_birth,
                            labels=labels, *args, **kwargs)

    samples.root = root

    return samples
