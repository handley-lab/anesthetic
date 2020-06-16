"""Generic sampling reading tools."""
import os
from anesthetic.read.getdistreader import GetDistReader
from anesthetic.read.montepythonreader import MontePythonReader
from anesthetic.read.polychordreader import PolyChordReader
from anesthetic.read.multinestreader import MultiNestReader


def SampleReader(root):
    """Read from a variety of sampling chains."""
    mn = MultiNestReader(root)
    if os.path.isfile(mn.ev_file) and os.path.isfile(mn.phys_live_file):
        return mn

    pc = PolyChordReader(root)
    if os.path.isfile(pc.birth_file):
        return pc

    gd = GetDistReader(root)
    if ((os.path.isfile(gd.paramnames_file) or os.path.isfile(gd.yaml_file))
            and os.path.isfile(gd.chains_files[0])
            and os.path.isfile(gd.chains_files[-1])):
        return gd

    mp = MontePythonReader(root)
    if (os.path.isfile(mp.log_param_file)
            and os.path.isfile(mp.log_file)
            and os.path.isfile(mp.paramnames_file)):
        mp._init()
        return mp

    errstr = "Could not find MCMC sampling chains with root %s" % root
    try:
        raise FileNotFoundError(errstr)
    except NameError:
        raise IOError(errstr)
