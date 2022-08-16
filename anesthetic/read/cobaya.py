"""Read MCMCSamples from cobaya chains."""
import os
import glob
import numpy as np
from anesthetic.read.utils import remove_burn_in
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
        paramnames = header.split()[2:]
        return paramnames


def read_cobaya(root, *args, **kwargs):
    """Read Cobaya yaml files.

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

    pattern = '.[0-9]*.txt'
    chains_files = sorted(glob.glob(root + pattern))
    if len(chains_files) > 1:
        chain_numbers = [cf.split('.')[-2] for cf in chains_files]
    else:
        chain_numbers = [0]
    if not chains_files:
        raise FileNotFoundError(root + pattern + " not found.")

    params = read_paramnames(root)
    columns = kwargs.pop('columns', params)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    samples = []
    for i, chains_file in enumerate(chains_files):
        data = np.loadtxt(chains_file)
        data = remove_burn_in(data, burn_in)
        chain = np.full((data.shape[0], 1), int(chain_numbers[i]))
        data = np.hstack((data, chain))
        samples.append(data)

    samples = np.vstack(samples)
    columns.append('chain')
    weights, logP, data = np.split(samples, [1, 2], axis=1)
    samples = MCMCSamples(data=data, columns=columns,
                          weights=weights.flatten(), logL=logP,
                          root=root, *args, **kwargs)
    samples.chain = samples.chain.astype(int)

    if np.all(samples.chain == 0):
        samples.drop('chain', inplace=True, axis=1)
    else:
        samples.tex['chain'] = r'$n_\mathrm{chain}$'

    return samples
