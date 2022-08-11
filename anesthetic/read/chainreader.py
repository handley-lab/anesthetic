"""Tools for reading from chains files."""


class ChainReader(object):
    """Base class for reading chains files."""

    def __init__(self, root):
        self.root = root

    def paramnames(self):
        """Parameter names mapping."""
        return None, {}

    def samples(self):
        """Read samples from file."""
        raise NotImplementedError
