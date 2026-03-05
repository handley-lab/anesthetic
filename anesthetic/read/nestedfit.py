"""Read NestedSamples from Nested_Fit chains."""
import os
import numpy as np
from anesthetic.read.getdist import read_paramnames
from anesthetic.samples import NestedSamples


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
    root_getdist = os.path.join(root, 'nf_output_points')
    columns, labels = read_paramnames(root_getdist)
    # No specific labeling is implemented in nested_fit
    labels = columns
    columns = kwargs.pop('columns', columns)
    labels = kwargs.pop('labels', labels)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    return NestedSamples(data=data, columns=columns,
                         logL=logL, logL_birth=logL_birth,
                         labels=labels, *args, **kwargs)
