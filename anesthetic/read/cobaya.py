import os
import glob
import numpy as np
from anesthetic.read.getdist import remove_burn_in
from anesthetic.samples import MCMCSamples


def read_paramnames(root):
    r"""Read header of <root>.1.txt to infer the paramnames.

    This is the data file of the first chain. It should have as many
    columns as there are parameters (sampled and derived) plus an
    additional two corresponding to the weights (first column) and the
    logposterior (second column). The first line should start with a # and
    should list the parameter names corresponding to the columns. This
    will be used as label in the pandas array.
    """
    with open(root + ".1.txt") as f:
        header = f.readline()[1:]
        paramnames = header.split()[2:-4]
        return paramnames


def read_cobaya(root, *args, **kwargs):
    """Read Cobaya yaml files."""
    burn_in = kwargs.pop('burn_in', None)

    pattern = '.[0-9]*.txt'
    chains_files = glob.glob(root + pattern)
    if not chains_files:
        raise FileNotFoundError(root + pattern  + " not found.")

    data = np.array([])
    for chains_file in chains_files:
        data_i = np.loadtxt(chains_file)
        data_i = remove_burn_in(data_i, burn_in)
        data = np.concatenate((data, data_i)) if data.size else data_i

    weights, logP, data, _ = np.split(data, [1, 2, -4], axis=1)
    params = read_paramnames(root)
    columns = kwargs.pop('columns', params)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    return MCMCSamples(data=data, columns=columns, weights=weights.flatten(),
                       logL=logP, root=root, *args, **kwargs)
