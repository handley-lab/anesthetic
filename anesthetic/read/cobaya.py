"""Read MCMCSamples from Cobaya chains."""
import os
import re
from anesthetic.read._utils import _read_mcmc_chains
from anesthetic.samples import MCMCSamples


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


def read_cobaya_paramnames(root):
    r"""Read parameter names and labels from a Cobaya chain header.

    ``<root>.1.txt`` is the data file of the first chain. It should have as
    many columns as there are parameters (sampled and derived) plus an
    additional two corresponding to the weights (first column) and the
    minus-log-posterior (second column). The first line should start with a
    ``#`` and should list the parameter names corresponding to the columns.
    These will be used as handles in the pandas array.

    Parameters
    ----------
    root : str
        Root name for reading Cobaya chain metadata.

    Returns
    -------
    parameters : list[str] or list[int]
        Parameter names in file order, excluding sampler bookkeeping fields.
    labels : dict
        Mapping from parameter names to axis labels.

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
        except ImportError:
            return paramnames, {}
        params = ParamNames(cobaya_params_file(root))
        labels = {p.name: '$' + p.label + '$' for p in params.names}
        for p in paramnames:
            if p == 'minuslogprior':
                labels.update({p: '$-\\ln\\pi$'})
            elif 'minuslogprior_' in p:
                sub = p.split('_', maxsplit=1)[-1].lstrip('_')
                labels.update({p: f'$-\\ln\\pi_\\mathrm{{{sub}}}$'})
        return paramnames, labels


def read_cobaya(root, *args, columns=None, renames=None, burn_in=None,
                thin=None, compress_repeats=False, **kwargs):
    """Read Cobaya chain files.

    When installed, `GetDist <https://getdist.readthedocs.io/en/latest/>`__
    is used to read parameter labels from Cobaya's YAML metadata.

    Parameters
    ----------
    root : str
        Root name for reading files in Cobaya format, i.e. the chain files
        ``<root>.<n>.txt`` and optional associated YAML metadata.

    columns : list[str], list[int], or slice, optional
        Optionally select which parameter columns to load from the chain files.
        This is useful when you do not want to load a large number of nuisance
        parameters into memory. Integer positions and slices index parameter
        fields only, not sampler bookkeeping fields such as ``logL``.

    renames : dict, optional
        Mapping from parameter names to new names (i.e. column handles).
        Labels are not carried over to renamed parameters, so provide them
        separately via a ``labels`` dict with the new parameter names as keys.

    burn_in : int, float or array-like, optional
        Number or fraction of stored rows to remove from each chain before
        loading samples into memory. Uses the same semantics as
        :meth:`anesthetic.samples.MCMCSamples.remove_burn_in`.

    thin : int, optional
        Keep every ``thin``-th sample in the expanded MCMC chain represented
        by the frequency weights.

    compress_repeats : bool, default=False
        Oversampling nuisance parameters can leave the selected parameters of
        interest unchanged across consecutive samples. Merge these repeated
        rows by summing their weights. Compression happens separately for each
        chain, after burn-in removal and thinning. If ``False``, likelihood
        bookkeeping fields such as ``logL`` are returned in addition to the
        selected columns. If ``True``, only the selected columns and ``chain``
        are returned. Weights are always retained.

    *args, **kwargs
        Passed on to ``MCMCSamples``. Check its docstring for more
        information.

    Returns
    -------
    :class:`anesthetic.samples.MCMCSamples`

    """
    dirname, basename = os.path.split(root)
    files = os.listdir(os.path.dirname(root))
    regex = re.escape(basename) + r'\.([0-9]+)\.txt$'
    matches = [re.match(regex, f) for f in files]
    chain_files = [(m.group(1), os.path.join(dirname, m.group(0)))
                   for m in matches if m]
    if not chain_files:
        raise FileNotFoundError(dirname + '/' + regex + " not found.")
    chain_files.sort(key=lambda chain_file: int(chain_file[0]))

    parameters, labels = read_cobaya_paramnames(root)

    data, columns, weights, minuslogP, chains, renames = _read_mcmc_chains(
        chain_files, parameters, columns, _count_samples,
        header_rows=1, burn_in=burn_in, thin=thin,
        compress_repeats=compress_repeats, renames=renames
    )
    logL = None if compress_repeats else -data[:, columns.index('chi2')] / 2

    columns = [renames.get(column, column) for column in columns]
    labels = kwargs.pop('labels', labels)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    samples = MCMCSamples(data=data, columns=columns,
                          weights=weights, logL=logL,
                          labels=labels, *args, **kwargs)

    if not compress_repeats:
        samples['logP'] = -minuslogP
        if samples.islabelled():
            samples.set_label('logP', '$\\ln\\mathcal{P}$')
    samples['chain'] = chains
    if samples.islabelled():
        samples.set_label('chain', r'$n_\mathrm{chain}$')
    samples.root = root

    return samples
