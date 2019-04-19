"""Generic nested sampling reading tools."""
import os
from anesthetic.read.polychord import PolyChordReader
from anesthetic.read.multinest import MultiNestReader


def NestedReader(root):
    """Read from a variety of nested sampling chains."""
    mn = MultiNestReader(root)
    if os.path.isfile(mn.ev_file) and os.path.isfile(mn.phys_live_file):
        return mn

    pc = PolyChordReader(root)
    if os.path.isfile(pc.birth_file):
        return pc

    errstr = "Could not find nested sampling chains with root %s" % root
    try:
        raise FileNotFoundError(errstr)
    except NameError:
        raise IOError(errstr)
