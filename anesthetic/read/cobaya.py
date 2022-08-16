"""Read MCMCSamples from cobaya chains."""
import os
import re
import numpy as np
from anesthetic.read.utils import remove_burn_in
from anesthetic.samples import MCMCSamples
from pandas import concat


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
        try:
            s = loadMCSamples(file_root=self.root)
            tex = {p.name: '$' + p.label + '$' for p in s.paramNames.names}
            return paramnames, tex
        except NameError:
            return paramnames, {}


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
    dirname, basename = os.path.split(root)

    files = os.listdir(os.path.dirname(root))
    regex = basename + r'.([0-9]+)\.txt'
    matches = [re.match(regex, f) for f in files]
    chains_files = [(m.group(1), os.path.join(dirname, m.group(0)))
                    for m in matches if m]
    if not chains_files:
        raise FileNotFoundError(dirname + '/' + regex + " not found.")

    params, tex = read_paramnames(root)
    columns = kwargs.pop('columns', params)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    samples = []
    for i, chains_file in chains_files:
        data = np.loadtxt(chains_file)
        data = remove_burn_in(data, burn_in)
        weights, logP, data = np.split(data, [1, 2], axis=1)
        mcmc = MCMCSamples(data=data, columns=columns,
                           weights=weights.flatten(), logL=logP, tex=tex,
                           root=root, *args, **kwargs)
        mcmc['chain'] = int(i) if i else np.nan
        samples.append(mcmc)

    samples = concat(samples)
    samples.sort_values(by=['chain', '#'], inplace=True)
    weights = samples.weights
    samples.reset_index(inplace=True, drop=True)
    samples.weights = weights

    if np.all(samples.chain == 1):
        samples.drop('chain', inplace=True, axis=1)
    else:
        samples.tex['chain'] = r'$n_\mathrm{chain}$'

    return samples
