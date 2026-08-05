"""Read NestedSamples from Nested_fit chains."""
import os
import numpy as np
from anesthetic.read.getdist import read_getdist_paramnames
from anesthetic.samples import NestedSamples


def read_nestedfit_paramnames(root):
    """Read parameter names and labels for Nested_fit chains."""
    root_getdist = os.path.join(root, 'nf_output_points')
    return read_getdist_paramnames(root_getdist)


def read_nestedfit(root, *args, **kwargs):
    """Read Nested_Fit chain files.

    Parameters
    ----------
    root : str
        root specify the directory only, no specific roots,
        The files read files are ``nf_output_points.txt``
        and ``nf_output_diag.txt``.

    """
    dead_file = os.path.join(root, 'nf_output_points.txt')
    birth_file = os.path.join(root, 'nf_output_diag.dat')
    data_dead = np.loadtxt(dead_file)
    data_birth = np.loadtxt(birth_file)
    weight, logL, data = np.split(data_dead, [1, 2], axis=1)
    logL_birth = data_birth[:, 0]
    columns, _ = read_nestedfit_paramnames(root)
    columns = kwargs.pop('columns', columns)
    # Nested_fit does not provide separate parameter labels.
    labels = kwargs.pop('labels', columns)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    return NestedSamples(data=data, columns=columns,
                         logL=logL, logL_birth=logL_birth,
                         labels=labels, *args, **kwargs)
