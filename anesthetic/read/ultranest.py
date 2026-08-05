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


def read_ultranest(root, *args, columns=None, **kwargs):
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

    *args, **kwargs
        Passed on to ``NestedSamples``. Check its docstring for more
        information.

    Returns
    -------
    :class:`anesthetic.samples.NestedSamples`

    """
    parameters, _ = read_ultranest_paramnames(root)
    num_params = len(parameters)
    indices, columns = _norm_columns(columns, parameters)

    filepath = os.path.join(root, 'results', 'points.hdf5')
    try:
        import h5py
    except ImportError:
        raise ImportError('h5py is required to read UltraNest results')
    with h5py.File(filepath, 'r') as fileobj:
        points = fileobj['points']
        _, ncols = points.shape
        x_dim = ncols - 3 - num_params
        logL_birth = points[:, 0]
        logL = points[:, 1]
        if indices is None:
            samples = points[:, 3+x_dim:3+x_dim+num_params]
        else:
            samples = points[:, [3+x_dim+i for i in indices]]

    kwargs['label'] = kwargs.get('label', os.path.basename(root))
    labels = kwargs.pop('labels', columns)
    data = samples

    return NestedSamples(data=data, logL=logL, logL_birth=logL_birth,
                         columns=columns, labels=labels, *args, **kwargs)
