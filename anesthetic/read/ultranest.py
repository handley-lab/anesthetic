"""Read NestedSamples from UltraNest results."""
import os
import json
from anesthetic.read._utils import _norm_columns
from anesthetic.samples import NestedSamples


def read_ultranest_paramnames(root):
    """Read parameter names and labels for UltraNest chains."""
    with open(os.path.join(root, 'info', 'results.json')) as infofile:
        parameters = json.load(infofile)['paramnames']
    return parameters, {}


def read_ultranest(root, *args, columns=None, renames=None, **kwargs):
    """Read UltraNest files.

    Parameters
    ----------
    root : str
        Root name for reading files in UltraNest format, i.e. the files
        ``<root>/info/results.json`` and ``<root>/results/points.hdf5``.

    columns : list[str], list[int], or slice, optional
        Optionally select which parameter columns to load from the chain files.
        This is useful when you do not want to load a large number of nuisance
        parameters into memory. Integer positions and slices index parameter
        fields only, not sampler bookkeeping fields such as ``logL``.

    renames : dict, optional
        Mapping from parameter names to new names (i.e. column handles).
        Labels are not carried over to renamed parameters, so provide them
        separately via a ``labels`` dict with the new parameter names as keys.

    *args, **kwargs
        Passed on to ``NestedSamples``. Check its docstring for more
        information.

    Returns
    -------
    :class:`anesthetic.samples.NestedSamples`

    """
    points_file = os.path.join(root, 'results', 'points.hdf5')
    if not os.path.exists(points_file):
        raise FileNotFoundError(f"{points_file} not found.")

    parameters, _ = read_ultranest_paramnames(root)
    nparams = len(parameters)
    indices, columns, renames = _norm_columns(columns, parameters, renames)

    try:
        import h5py
    except ImportError:
        raise ImportError('h5py is required to read UltraNest results')
    with h5py.File(points_file, 'r') as fileobj:
        points = fileobj['points']
        _, ncols = points.shape
        x_dim = ncols - 3 - nparams
        logL_birth = points[:, 0]
        logL = points[:, 1]
        if indices is None:
            data = points[:, 3+x_dim:3+x_dim+nparams]
        else:
            data = points[:, [3+x_dim+i for i in indices]]

    columns = [renames.get(column, column) for column in columns]
    # UltraNest does not provide separate parameter labels.
    labels = kwargs.pop('labels', columns)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    samples = NestedSamples(data=data, columns=columns,
                            logL=logL, logL_birth=logL_birth,
                            labels=labels, *args, **kwargs)

    samples.root = root

    return samples
