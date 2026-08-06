"""Read MCMCSamples from GetDist chains."""
import os
import re
import warnings
import numpy as np
from anesthetic.read._utils import _read_mcmc_chains
from anesthetic.samples import MCMCSamples


def _count_samples(filename):
    """Count samples in a GetDist chain file."""
    with open(filename) as file:
        return sum(bool(line.strip()) and not line.lstrip().startswith('#')
                   for line in file)


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
    parameters : list[str] or list[int]
        Parameter names in file order, excluding sampler bookkeeping fields.
    labels : dict
        Mapping from parameter names to axis labels.

    """
    paramnames_file = root + '.paramnames'
    try:
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


def read_getdist(root, *args, columns=None, renames=None, burn_in=None,
                 thin=None, compress_repeats=False, **kwargs):
    """Read GetDist chain files.

    Parameters
    ----------
    root : str
        Root name for reading files in GetDist format, i.e. the chain files
        and optional ``<root>.paramnames`` file.

    columns : list[str], list[int], or slice, optional
        Optionally select which parameter columns to load from the chain files.
        This is useful when you do not want to load a large number of nuisance
        parameters into memory. Integer positions and slices index parameter
        fields only, not sampler bookkeeping fields such as ``logL``.

    renames : dict, optional
        Mapping from parameter names to new names.

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
    regex = re.escape(basename) + r'(?:(?:_|\.)([0-9]+))?\.txt$'
    matches = [re.match(regex, f) for f in files]
    chain_files = [(m.group(1), os.path.join(dirname, m.group(0)))
                   for m in matches if m]
    if not chain_files:
        raise FileNotFoundError(dirname + '/' + regex + " not found.")
    chain_files.sort(key=lambda chain_file: int(chain_file[0] or 0))

    parameters, labels = read_getdist_paramnames(root)
    labels = kwargs.pop('labels', labels)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    data, columns, weights, minuslogL, chains, renames = _read_mcmc_chains(
        chain_files, parameters, columns, _count_samples,
        header_rows=0, burn_in=burn_in, thin=thin,
        compress_repeats=compress_repeats, renames=renames
    )

    logL = None if compress_repeats else -minuslogL
    columns = [renames.get(column, column) for column in columns]
    samples = MCMCSamples(data=data, columns=columns, weights=weights,
                          logL=logL, labels=labels, *args, **kwargs)
    samples['chain'] = chains
    if samples.islabelled():
        samples.set_label('chain', r'$n_\mathrm{chain}$')
    samples.root = root
    samples.label = kwargs['label']

    return samples
