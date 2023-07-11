"""Read NestedSamples from ultranest results."""
import os
from anesthetic.samples import NestedSamples
import h5py
import json

def read_ultranest(root, *args, **kwargs):
    """Read <root>ev.dat and <root>phys_live.points in MultiNest format."""
    labels = json.load(open(os.path.join(root, 'info', 'results.json')))['paramnames']
    filepath = os.path.join(root, 'results', 'points.hdf5')
    fileobj = h5py.File(filepath, 'r')
    points = fileobj['points']
    _, ncols = points.shape
    # assume the transform did not add parameters, then x_dim == num_params
    num_params = len(labels)
    x_dim = ncols - 3 - num_params
    logL_birth = points[:,0]
    logL = points[:,1]
    # u_samples = points[:,3:x_dim]
    samples = points[:,3+x_dim:3+x_dim+num_params]

    kwargs['label'] = kwargs.get('label', os.path.basename(root))
    columns = kwargs.pop('columns', labels)
    labels = kwargs.pop('labels', labels)
    data = samples

    return NestedSamples(data=data, logL=logL, logL_birth=logL_birth,
                         columns=columns, labels=labels, *args, **kwargs)
