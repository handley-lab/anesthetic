from pandas.io.pytables import HDFStore as _HDFStore
from anesthetic import __version__
from anesthetic.samples import NestedSamples, MCMCSamples, Samples


class HDFStore(_HDFStore):
    anesthetic_types = {x.__name__: x
                        for x in [NestedSamples, MCMCSamples, Samples]}

    def get(self, key):
        storer = self.get_storer(key)
        anesthetic_type = storer.attrs.anesthetic_type
        anesthetic_type = self.anesthetic_types[anesthetic_type]
        value = super().get(key)
        value = anesthetic_type(value)
        _metadata = storer.attrs._metadata.keys()
        value._metadata = list(_metadata)
        for k, v in storer.attrs._metadata.items():
            setattr(value, k, v)
        return value

    def put(self, key, value):
        super().put(key, value)
        storer = self.get_storer(key)
        storer.attrs._metadata = {
                k: getattr(value,k)
                for k in value._metadata
                }
        storer.attrs.anesthetic_type = type(value).__name__
        storer.attrs.anesthetic_version = __version__
