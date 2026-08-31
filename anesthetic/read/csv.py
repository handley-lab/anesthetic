"""Read and write CSV files for anesthetic."""
from anesthetic.read._utils import _infer_weight_dtype
from anesthetic.weighted_labelled_pandas import read_csv as wl_read_csv
from anesthetic.samples import MCMCSamples, NestedSamples
from pathlib import Path


def read_csv(filename, *args, **kwargs):
    """Read a CSV file into a :class:`anesthetic.samples.Samples` object."""
    root = filename
    try:
        filename = Path(filename)
        kwargs['label'] = kwargs.get('label', filename.stem)
        filename = filename.with_suffix('.csv')
    except TypeError:
        pass
    wldf = wl_read_csv(filename)
    if 'nlive' in wldf.columns:
        samples = NestedSamples(wldf, *args, **kwargs)
    else:
        wldf.set_weights(_infer_weight_dtype(wldf.get_weights()), inplace=True)
        samples = MCMCSamples(wldf, *args, **kwargs)

    samples.root = root

    return samples
