"""Read MCMCSamples from Cobaya chains."""
import os
import re
import numpy as np
from anesthetic.samples import MCMCSamples
from pandas import concat


def read_paramnames(root):
    """Read header of ``<root>.1.txt`` to infer the paramnames.

    This is the data file of the first chain. It should have as many
    columns as there are parameters (sampled and derived) plus an
    additional two corresponding to the weights (first column) and the
    log-posterior (second column). The first line should start with a # and
    should list the parameter names corresponding to the columns. These
    will be used as handles in the pandas array.
    """
    with open(root + ".1.txt") as f:
        header = f.readline()[1:]
        paramnames = header.split()[2:]
        try:
            from getdist import loadMCSamples
            s = loadMCSamples(file_root=root)
            labels = {p.name: '$' + p.label + '$' for p in s.paramNames.names}
            return paramnames, labels
        except ImportError:
            return paramnames, {}


def read_cobaya(root, *args, **kwargs):
    """Read Cobaya yaml files.

    Note that in order to optimally read chains from Cobaya you need to have
    `GetDist <https://getdist.readthedocs.io/en/latest/>`__ installed.

    Parameters
    ----------
    root : str
        root name for reading files in Cobaya format, i.e. the files
        ``<root>.*.txt`` and ``<root>.updated.yaml``.

    Returns
    -------
    :class:`anesthetic.samples.MCMCSamples`

    """
    dirname, basename = os.path.split(root)

    files = os.listdir(os.path.dirname(root))
    regex = re.escape(basename) + r'.([0-9]+)\.txt'
    matches = [re.match(regex, f) for f in files]
    chains_files = [(m.group(1), os.path.join(dirname, m.group(0)))
                    for m in matches if m]
    if not chains_files:
        raise FileNotFoundError(dirname + '/' + regex + " not found.")

    columns, labels = read_paramnames(root)
    columns = kwargs.pop('columns', columns)
    labels = kwargs.pop('labels', labels)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    samples = []
    for i, chains_file in chains_files:
        data = np.loadtxt(chains_file)
        weights, logP, data = np.split(data, [1, 2], axis=1)
        mcmc = MCMCSamples(data=data, columns=columns,
                           weights=weights.flatten(), logL=logP,
                           labels=labels, *args, **kwargs)
        mcmc['chain'] = int(i) if i else np.nan
        samples.append(mcmc)

    samples = concat(samples)
    samples.index.names = ['index', 'weights']
    samples.sort_values(by=['chain', 'index'], inplace=True)
    samples.reset_index(inplace=True, drop=True)
    samples.root = root
    samples.label = kwargs['label']

    if np.all(samples.chain == samples.chain.iloc[0]):
        samples.drop(columns='chain', inplace=True, level=0)
    else:
        samples.set_label('chain', r'$n_\mathrm{chain}$')

    return samples
