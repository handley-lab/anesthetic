"""Read NestedSamples from UltraNest results."""
import os
import json
from anesthetic.samples import NestedSamples


def read_ultranest(root, *args, **kwargs):
    """Read UltraNest files.

    Parameters
    ----------
    root : str
        root name for reading files in UltraNest format, i.e. the files
        ``<root>/info/results.json`` and ``<root>/results/points.hdf5``.

    """
    with open(os.path.join(root, 'info', 'results.json')) as infofile:
        labels = json.load(infofile)['paramnames']
    num_params = len(labels)

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
        samples = points[:, 3+x_dim:3+x_dim+num_params]

    kwargs['label'] = kwargs.get('label', os.path.basename(root))
    columns = kwargs.pop('columns', labels)
    labels = kwargs.pop('labels', labels)
    data = samples

    return NestedSamples(data=data, logL=logL, logL_birth=logL_birth,
                         columns=columns, labels=labels, *args, **kwargs)
