"""Tools for reading from chains files."""

class ChainReader(object):
    """Base class for reading chains files."""
    def __init__(self, root):
        """Save file root for loading files"""
        self.root = root

    def paramnames(self):
        """Default parameter names mapping"""
        return None, None
    
    def limits(self):
        """Default parameter limits mapping"""
        return {}

    def samples(self):
        """Default samples"""
        raise NotImplementedError
