"""Read NestedSamples from Nested_fit chains."""
import os
import numpy as np
from anesthetic.read._utils import _norm_columns
from anesthetic.read.getdist import read_getdist_paramnames
from anesthetic.samples import NestedSamples


def read_nestedfit_paramnames(root):
    """Read parameter names and labels for Nested_fit chains."""
    root_getdist = os.path.join(root, 'nf_output_points')
    return read_getdist_paramnames(root_getdist)


def read_nestedfit(root, *args, columns=None, **kwargs):
    """Read Nested_fit chain files.

    Parameters
    ----------
    root : str
        Root directory containing ``nf_output_points.txt`` and
        ``nf_output_diag.dat``.

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
    dead_file = os.path.join(root, 'nf_output_points.txt')
    birth_file = os.path.join(root, 'nf_output_diag.dat')
    parameters, _ = read_nestedfit_paramnames(root)
    indices, columns = _norm_columns(columns, parameters)
    usecols = None if indices is None else [0, 1] + [i+2 for i in indices]

    data_dead = np.loadtxt(dead_file, usecols=usecols, ndmin=2)
    weight, logL, data = np.split(data_dead, [1, 2], axis=1)
    logL_birth = np.loadtxt(birth_file, usecols=0, ndmin=1)
    # Nested_fit does not provide separate parameter labels.
    labels = kwargs.pop('labels', columns)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    return NestedSamples(data=data, columns=columns,
                         logL=logL, logL_birth=logL_birth,
                         labels=labels, *args, **kwargs)
