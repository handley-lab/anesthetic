"""Anesthetic overwrites for pandas hdf functionality."""
from pandas import HDFStore as _HDFStore
from pandas.io.pytables import to_hdf as _to_hdf, read_hdf as _read_hdf
from anesthetic.utils import adjust_docstrings
from anesthetic.samples import NestedSamples, MCMCSamples, Samples


class HDFStore(_HDFStore):  # noqa: D101
    anesthetic_types = {x.__name__: x
                        for x in [NestedSamples, MCMCSamples, Samples]}

    def get(self, key, *args, **kwargs):  # noqa: D102
        storer = self.get_storer(key)
        anesthetic_type = storer.attrs.anesthetic_type
        anesthetic_type = self.anesthetic_types[anesthetic_type]
        value = super().get(key, *args, **kwargs)
        value = anesthetic_type(value)
        _metadata = storer.attrs._metadata.keys()
        value._metadata = list(_metadata)
        for k, v in storer.attrs._metadata.items():
            setattr(value, k, v)
        return value

    def put(self, key, value, *args, **kwargs):  # noqa: D102
        from anesthetic import __version__
        super().put(key, value, *args, **kwargs)
        storer = self.get_storer(key)
        storer.attrs._metadata = {
                k: getattr(value, k)
                for k in value._metadata
                }
        storer.attrs.anesthetic_type = type(value).__name__
        storer.attrs.anesthetic_version = __version__

    def select(self, key, *args, **kwargs):  # noqa: D102
        storer = self.get_storer(key)
        anesthetic_type = storer.attrs.anesthetic_type
        anesthetic_type = self.anesthetic_types[anesthetic_type]
        value = super().select(key, *args, **kwargs)
        value = anesthetic_type(value)
        _metadata = storer.attrs._metadata.keys()
        value._metadata = list(_metadata)
        for k, v in storer.attrs._metadata.items():
            setattr(value, k, v)
        return value


def to_hdf(path_or_buf, key, value, mode="a", complevel=None, complib=None,
           *args, **kwargs):  # noqa: D103

    store = HDFStore(path_or_buf, mode=mode, complevel=complevel,
                     complib=complib)
    store.__fspath__ = lambda: store
    return _to_hdf(store, key, value, *args, **kwargs)


def read_hdf(path_or_buf, *args, **kwargs):  # noqa: D103
    store = HDFStore(path_or_buf)
    return _read_hdf(store, *args, **kwargs)


to_hdf.__doc__ = _to_hdf.__doc__
read_hdf.__doc__ = _read_hdf.__doc__
adjust_docstrings(read_hdf, 'read_hdf', 'anesthetic.read_hdf')
adjust_docstrings(read_hdf, 'DataFrame', 'pandas.DataFrame')
adjust_docstrings(read_hdf, ':func:`open`', '`open`')
adjust_docstrings(read_hdf, ':class:`pandas.HDFStore`', '`pandas.HDFStore`')
