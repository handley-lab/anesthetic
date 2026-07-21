"""Read MCMCSamples from Cobaya chains."""
from itertools import compress, islice
import os
import re
import numpy as np
from anesthetic.samples import MCMCSamples, _compute_burn_in, _thin_weights


def _count_samples(filename):
    """Count samples in a Cobaya chain file."""
    with open(filename) as f:
        return sum(bool(line.strip()) and not line.lstrip().startswith('#')
                   for line in f)


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
        header = f.readline().lstrip()
        fields = header[1:].split() if header.startswith('#') else []
        if fields[:2] != ['weight', 'minuslogpost']:
            raise IOError(root + ".1.txt has no Cobaya chain header.")
        paramnames = fields[2:]
        try:
            from getdist.cobaya_interface import cobaya_params_file
            from getdist.paramnames import ParamNames
            params = ParamNames(cobaya_params_file(root))
            labels = {p.name: '$' + p.label + '$' for p in params.names}
            for p in paramnames:
                if p == 'minuslogprior':
                    labels.update({p: '$-\\ln\\pi$'})
                elif 'minuslogprior_' in p:
                    sub = p.split('_', maxsplit=1)[-1].lstrip('_')
                    labels.update({p: f'$-\\ln\\pi_\\mathrm{{{sub}}}$'})
            return paramnames, labels
        except ImportError:
            return paramnames, {}


def read_cobaya(root, *args, burn_in=None, thin=None, **kwargs):
    """Read Cobaya yaml files.

    Note that in order to optimally read chains from Cobaya you need to have
    `GetDist <https://getdist.readthedocs.io/en/latest/>`__ installed.

    Parameters
    ----------
    root : str
        root name for reading files in Cobaya format, i.e. the files
        ``<root>.*.txt`` and ``<root>.updated.yaml``.

    burn_in : int, float or array-like, optional
        Number or fraction of stored rows to remove from each chain before
        loading samples into memory. Uses the same semantics as
        :meth:`anesthetic.samples.MCMCSamples.remove_burn_in`.

    thin : int, optional
        Keep every ``thin``-th sample in the expanded MCMC chain represented
        by the integer weights.

    Returns
    -------
    :class:`anesthetic.samples.MCMCSamples`

    """
    dirname, basename = os.path.split(root)

    files = os.listdir(os.path.dirname(root))
    regex = re.escape(basename) + r'.([0-9]+)\.txt'
    matches = [re.match(regex, f) for f in files]
    chain_files = [(m.group(1), os.path.join(dirname, m.group(0)))
                   for m in matches if m]
    if not chain_files:
        raise FileNotFoundError(dirname + '/' + regex + " not found.")
    chain_files.sort(key=lambda chain_file: int(chain_file[0]))

    columns, labels = read_paramnames(root)
    columns = kwargs.pop('columns', columns)
    labels = kwargs.pop('labels', labels)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    chain_lengths = np.array([_count_samples(file)
                              for _, file in chain_files])
    if burn_in is None:
        ndrop = np.zeros(len(chain_lengths), dtype=int)
    else:
        ndrop = _compute_burn_in(burn_in, chain_lengths)
    selected_lengths = chain_lengths - ndrop
    selected_weights = []
    if thin is not None:
        for j, ((_, chain_file), skip, selected) in enumerate(zip(
                chain_files, ndrop, selected_lengths)):
            selected_weights.append(_thin_weights(
                np.loadtxt(
                    chain_file, skiprows=skip+1, usecols=0, dtype=int, ndmin=1
                ),
                thin
            ))
            selected_lengths[j] = np.count_nonzero(selected_weights[-1])

    nsamples = sum(selected_lengths)
    data = np.empty((nsamples, len(columns)))
    weights = np.empty(nsamples, dtype=int)
    minuslogP = np.empty(nsamples)
    chains = np.empty(nsamples, dtype=int)

    start = 0
    for j, ((i, chain_file), skip, selected) in enumerate(zip(
            chain_files, ndrop, selected_lengths)):
        stop = start + selected
        if thin is None:
            chain_data = np.loadtxt(chain_file, skiprows=skip+1, ndmin=2)
            weights[start:stop] = chain_data[:, 0]
        else:
            mask = selected_weights[j] > 0
            with open(chain_file) as file:
                lines = compress(islice(file, skip+1, None), mask)
                chain_data = np.loadtxt(lines, ndmin=2)
            weights[start:stop] = selected_weights[j][mask]
        minuslogP[start:stop] = chain_data[:, 1]
        data[start:stop] = chain_data[:, 2:]
        chains[start:stop] = int(i) if i else np.nan
        start = stop

    samples = MCMCSamples(data=data, columns=columns, weights=weights,
                          labels=labels, *args, **kwargs)
    samples['logP'] = -minuslogP
    samples.set_label('logP', '$\\ln\\mathcal{P}$')
    samples['logL'] = -samples['chi2'] / 2
    samples.set_label('logL', '$\\ln\\mathcal{L}$')
    samples['chain'] = chains
    samples.set_label('chain', r'$n_\mathrm{chain}$')
    samples.root = root
    samples.label = kwargs['label']

    return samples
