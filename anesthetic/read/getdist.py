"""Read MCMCSamples from GetDist chains."""
import os
import re
import numpy as np
from anesthetic.samples import MCMCSamples
from pandas import concat


def read_paramnames(root):
    r"""Read ``<root>.paramnames`` in GetDist format.

    This file should contain one or two columns. The first column indicates
    a reference name for the sample, used as labels in the pandas array.
    The second optional column should include the equivalent axis label,
    possibly in tex, with the understanding that it will be surrounded by
    dollar signs, for example

    ``<root>.paramnames``:
    ::

        a1     a_1
        a2     a_2
        omega  \omega

    """
    try:
        paramnames_file = root + '.paramnames'
        with open(paramnames_file, 'r') as f:
            paramnames = []
            labels = {}
            for line in f:
                line = line.strip().split()
                paramname = line[0].replace('*', '')
                paramnames.append(paramname)
                if len(line) > 1:
                    labels[paramname] = '$' + ' '.join(line[1:]) + '$'
            return paramnames, labels
    except IOError:
        return None, {}


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

    columns, labels = read_paramnames(root)
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
        mcmc['chain'] = int(i) if i else np.nan
        samples.append(mcmc)

    samples = concat(samples)
    samples.index.names = ['index', 'weights']
    samples.sort_values(by=['chain', 'index'], inplace=True)
    samples.reset_index(inplace=True, drop=True)
    samples.root = root
    samples.label = kwargs['label']

    all_same_chain = np.all(samples.chain == samples.chain.iloc[0])
    if all_same_chain or samples.chain.isna().all():
        samples.drop(columns='chain', inplace=True, level=0)
    elif samples.islabelled():
        samples.set_label('chain', r'$n_\mathrm{chain}$')

    return samples
