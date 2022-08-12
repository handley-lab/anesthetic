import os
import glob
import numpy as np
from anesthetic.samples import MCMCSamples

def read_paramnames(root):
    r"""Read <root>.paramnames in getdist format.

    This file should contain one or two columns. The first column indicates
    a reference name for the sample, used as labels in the pandas array.
    The second optional column should include the equivalent axis label,
    possibly in tex, with the understanding that it will be surrounded by
    dollar signs, for example

    <root.paramnames>

    a1     a_1
    a2     a_2
    omega  \omega
    """
    try:
        paramnames_file = root + '.paramnames'
        with open(paramnames_file, 'r') as f:
            paramnames = []
            tex = {}
            for line in f:
                line = line.strip().split()
                paramname = line[0].replace('*', '')
                paramnames.append(paramname)
                if len(line) > 1:
                    tex[paramname] = '$' + ' '.join(line[1:]) + '$'
            return paramnames, tex
    except IOError:
        return None, {}


def remove_burn_in(data, burn_in):
    """Strip burn in from a dataset."""
    if burn_in:
        if 0 < burn_in < 1:
            burn_in *= len(data)
        return data[np.ceil(burn_in).astype(int):]
    return data


def read_getdist(root, *args, **kwargs):
    """Read <root>_1.txt in getdist format.

    Parameters
    ----------

    burn_in: float
        if 0 < burn_in < 1:
            discard the first burn_in fraction of samples
        elif 1 < burn_in:
            only keep samples [burn_in:]
        Only works if `root` provided and if chains are GetDist or Cobaya
        compatible.
        default: False

    Returns
    -------
    MCMCSamples

    """
    burn_in = kwargs.pop('burn_in', None)

    chains_files = glob.glob(root + '_[0-9].txt')
    if not chains_files:
        chains_files = glob.glob(root + '.[0-9].txt')
    if not chains_files:
        chains_files = [root + '.txt']

    data = np.array([])
    for chains_file in chains_files:
        data_i = np.loadtxt(chains_file)
        data_i = remove_burn_in(data_i, burn_in)
    data = np.concatenate((data, data_i)) if data.size else data_i
    weights, minuslogL, data = np.split(data, [1, 2], axis=1)

    params, tex = read_paramnames(root)
    columns = kwargs.pop('columns', params)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    return MCMCSamples(data=data, columns=columns, weights=weights.flatten(),
                       logL=-minuslogL, tex=tex, root=root, *args, **kwargs)
