"""Read NestedSamples from ultranest results."""
import os
from anesthetic.samples import NestedSamples
import json


def read_ultranest(root, *args, **kwargs):
    """Read <root>ev.dat and <root>phys_live.points in MultiNest format."""
    import h5py

    with open(os.path.join(root, 'info', 'results.json')) as infofile:
        labels = json.load(infofile)['paramnames']
    num_params = len(labels)

    filepath = os.path.join(root, 'results', 'points.hdf5')
    with h5py.File(filepath, 'r') as fileobj:
        points = fileobj['points']
        _, ncols = points.shape
        x_dim = ncols - 3 - num_params
        logL_birth = points[:, 0]
        logL = points[:, 1]
        # u_samples = points[:,3:x_dim]
        samples = points[:, 3 + x_dim:3 + x_dim + num_params]

    kwargs['label'] = kwargs.get('label', os.path.basename(root))
    columns = kwargs.pop('columns', labels)
    labels = kwargs.pop('labels', labels)
    data = samples

    return NestedSamples(data=data, logL=logL, logL_birth=logL_birth,
                         columns=columns, labels=labels, *args, **kwargs)
