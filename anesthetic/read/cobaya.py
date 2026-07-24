"""Read MCMCSamples from Cobaya chains."""
from itertools import compress, islice
import os
import re
import numpy as np
from anesthetic.read._utils import normalise_columns
from anesthetic.samples import (MCMCSamples, _compute_burn_in, _thin_weights,
                                _compress_consecutive_duplicates)


def _count_samples(filename):
    """Count samples in a Cobaya chain file."""
    with open(filename, 'rb') as file:
        header = file.readline()
        first = file.readline()
        if not first:
            return 0

        row_size = len(first)
        data_size = os.fstat(file.fileno()).st_size - len(header)
        nrows, remainder = divmod(data_size, row_size)

        # Cobaya writes fixed-width rows, so the file size usually suffices.
        if remainder == 0:
            for i in {0, nrows // 2, nrows - 1}:
                file.seek(len(header) + i * row_size)
                row = file.read(row_size)
                if row.count(b'\n') != 1 or not row.endswith(b'\n'):
                    break
            else:
                return nrows

    # Fall back to checking each row for non-standard Cobaya files.
    with open(filename) as file:
        return sum(bool(line.strip()) and not line.lstrip().startswith('#')
                   for line in file)


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


def read_cobaya(root, *args, columns=None, burn_in=None, thin=None,
                compress_consecutive_duplicates=False, **kwargs):
    """Read Cobaya yaml files.

    Note that in order to optimally read chains from Cobaya you need to have
    `GetDist <https://getdist.readthedocs.io/en/latest/>`__ installed.

    Parameters
    ----------
    root : str
        root name for reading files in Cobaya format, i.e. the files
        ``<root>.*.txt`` and ``<root>.updated.yaml``.

    columns : list[str], list[int], or slice, optional
        Optionally select which parameter columns to load from the chain files.
        This is useful when you do not want to load a large number of nuisance
        parameters into memory. Integer positions and slices index the
        parameter names returned by
        :func:`anesthetic.read.chain.read_parameters`, not the leading weight
        and minus-log-posterior columns in the chain file. Weights are always
        included.

    burn_in : int, float or array-like, optional
        Number or fraction of stored rows to remove from each chain before
        loading samples into memory. Uses the same semantics as
        :meth:`anesthetic.samples.MCMCSamples.remove_burn_in`.

    thin : int, optional
        Keep every ``thin``-th sample in the expanded MCMC chain represented
        by the integer weights.

    compress_consecutive_duplicates : bool, default=False
        Oversampling nuisance parameters can leave the selected parameters of
        interest unchanged across consecutive samples. Merge these repeated
        rows by summing their weights. Compression happens separately for each
        chain, after burn-in removal and thinning. If ``False``, ``chi2`` is
        loaded and ``logP`` and ``logL`` are calculated in addition to the
        selected columns. If ``True``, only the selected columns and ``chain``
        are returned.

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

    parameters, labels = read_paramnames(root)
    column_indices, columns = normalise_columns(columns, parameters)
    labels = kwargs.pop('labels', labels)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    if column_indices is None and compress_consecutive_duplicates:
        usecols = [0] + list(range(2, len(parameters) + 2))
    elif column_indices is None:
        usecols = None
    elif compress_consecutive_duplicates:
        usecols = [0] + [index + 2 for index in column_indices]
    else:
        # need to include `chi2` for `logL` computation
        chi2_index = parameters.index('chi2')
        if not any(index == chi2_index and column == 'chi2'
                   for index, column in zip(column_indices, columns)):
            column_indices = column_indices + [chi2_index]
            columns = columns + ['chi2']
        usecols = [0, 1] + [index + 2 for index in column_indices]

    chain_lengths = np.array([_count_samples(file)
                              for _, file in chain_files])
    if burn_in is None:
        ndrop = np.zeros(len(chain_lengths), dtype=int)
    else:
        ndrop = _compute_burn_in(burn_in, chain_lengths)
    selected_lengths = chain_lengths - ndrop
    selected_weights = []
    # Pre-compute thinned weights to determine the required allocation size.
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

    # Preallocate for all post-thinning rows; compression may use fewer rows.
    nsamples = sum(selected_lengths)
    data = np.empty((nsamples, len(columns)))
    weights = np.empty(nsamples, dtype=int)
    minuslogP = np.empty(nsamples)
    chains = np.empty(nsamples, dtype=int)

    start = 0
    for j, ((i, chain_file), skip, selected) in enumerate(zip(
            chain_files, ndrop, selected_lengths)):
        if selected == 0:
            continue

        stop = start + selected
        if thin is None:
            # Fast path: load each chain in a single pass.
            chain_data = np.loadtxt(chain_file, skiprows=skip+1,
                                    usecols=usecols, ndmin=2)
            weights[start:stop] = chain_data[:, 0]
        else:
            # Load only the rows retained by the preceding weights-only pass.
            mask = selected_weights[j] > 0
            with open(chain_file) as file:
                lines = compress(islice(file, skip+1, None), mask)
                chain_data = np.loadtxt(lines, usecols=usecols, ndmin=2)
            weights[start:stop] = selected_weights[j][mask]

        if compress_consecutive_duplicates:
            indices, compressed_weights = (
                _compress_consecutive_duplicates(
                    chain_data[:, 1:], weights[start:stop]
                )
            )
            compressed_data = chain_data[indices, 1:]
            stop = start + len(compressed_data)
            data[start:stop] = compressed_data
            weights[start:stop] = compressed_weights
        else:
            minuslogP[start:stop] = chain_data[:, 1]
            data[start:stop] = chain_data[:, 2:]

        chains[start:stop] = int(i)
        start = stop

    # Remove rows left unused by compression.
    if start < nsamples:
        data.resize((start, len(columns)))
        weights.resize(start)
        minuslogP.resize(start)
        chains.resize(start)

    samples = MCMCSamples(data=data, columns=columns, weights=weights,
                          labels=labels, *args, **kwargs)
    if not compress_consecutive_duplicates:
        samples['logP'] = -minuslogP
        samples.set_label('logP', '$\\ln\\mathcal{P}$')
        samples['logL'] = -data[:, columns.index('chi2')] / 2
        samples.set_label('logL', '$\\ln\\mathcal{L}$')
    samples['chain'] = chains
    samples.set_label('chain', r'$n_\mathrm{chain}$')
    samples.root = root
    samples.label = kwargs['label']

    return samples
