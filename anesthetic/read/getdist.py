"""Read MCMCSamples from GetDist chains."""
import os
import re
import warnings
import numpy as np
from anesthetic.samples import MCMCSamples
from pandas import concat


def read_getdist_paramnames(root):
    r"""Read parameter names and labels for GetDist-compatible chains.

    ``<root>.paramnames`` should contain one or two columns.
    The first column gives the parameter name used as a column name in the
    pandas array. The second optional column gives the corresponding axis
    label, possibly in TeX, with the understanding that it will be surrounded
    by dollar signs, for example

    ``<root>.paramnames``:
    ::

        a1     a_1
        a2     a_2
        omega  \omega

    Parameters
    ----------
    root : str
        Root name for reading GetDist-compatible chain metadata.

    Returns
    -------
    parameters : list of str
        Parameter names in file order, excluding sampler bookkeeping fields.
    labels : dict
        Mapping from parameter names to axis labels.

    """
    try:
        paramnames_file = root + '.paramnames'
        with open(paramnames_file, 'r', encoding='utf-8-sig') as f:
            paramnames = []
            labels = {}
            for line in f:
                line = line.strip().split(maxsplit=1)
                paramname = line[0].replace('*', '')
                paramnames.append(paramname)
                if len(line) > 1:
                    labels[paramname] = f"${line[1]}$"
            return paramnames, labels
    except IOError:
        pass

    if os.path.exists(root + '.txt'):
        chain_file = root + '.txt'
        nbookkeeping = 2
    elif os.path.exists(root + '_1.txt'):
        chain_file = root + '_1.txt'
        nbookkeeping = 2
    elif os.path.exists(root + '.1.txt'):
        chain_file = root + '.1.txt'
        nbookkeeping = 2
    elif os.path.exists(root + '_dead-birth.txt'):
        chain_file = root + '_dead-birth.txt'
        nbookkeeping = 2
    elif os.path.exists(root + 'dead-birth.txt'):
        chain_file = root + 'dead-birth.txt'
        nbookkeeping = 4
    elif os.path.exists(root + 'ev.dat'):
        chain_file = root + 'ev.dat'
        nbookkeeping = 3
    else:
        raise FileNotFoundError(f"No parameter metadata or supported chain "
                                f"file found for {root}.")

    nparams = np.loadtxt(chain_file, max_rows=1, ndmin=1).size - nbookkeeping
    warnings.warn(f"{paramnames_file} not found. Using integer parameter "
                  f"names inferred from {chain_file}.")
    return list(range(nparams)), {}


def read_getdist(root, *args, **kwargs):
    """Read <root>_1.txt in GetDist format.

    Returns
    -------
    :class:`anesthetic.samples.MCMCSamples`

    """
    dirname, basename = os.path.split(root)
    files = os.listdir(os.path.dirname(root))
    regex = re.escape(basename) + r'((_|.)([0-9]+)|)\.txt'
    matches = [re.match(regex, f) for f in files]
    chains_files = [(m.group(3), os.path.join(dirname, m.group(0)))
                    for m in matches if m]
    if not chains_files:
        raise FileNotFoundError(dirname + '/' + regex + " not found.")

    columns, labels = read_getdist_paramnames(root)
    columns = kwargs.pop('columns', columns)
    labels = kwargs.pop('labels', labels)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    samples = []
    for i, chains_file in chains_files:
        data = np.loadtxt(chains_file)
        weights, minuslogL, data = np.split(data, [1, 2], axis=1)
        mcmc = MCMCSamples(data=data, columns=columns,
                           weights=weights.flatten(), logL=-minuslogL,
                           labels=labels, *args, **kwargs)
        mcmc['chain'] = int(i or 0)
        samples.append(mcmc)

    samples = concat(samples)
    samples.index.names = ['index', 'weights']
    samples.sort_values(by=['chain', 'index'], inplace=True)
    samples.reset_index(inplace=True, drop=True)
    samples.root = root
    samples.label = kwargs['label']

    if samples.islabelled():
        samples.set_label('chain', r'$n_\mathrm{chain}$')

    return samples
