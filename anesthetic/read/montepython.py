"""Tools for reading from MontePython chains files."""
import sys
import os
import warnings
import numpy as np
from anesthetic.samples import MCMCSamples
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from montepython import analyze
except ImportError:
    pass


def read_montepython(root, *args, **kwargs):
    """Read and process MontePython chain files using `montepython.analyze`.

    Uses MontePython's `analyze` module to prepare the data:
        * Extracts parameter names from MontePython's log.param file.
        * Removes burn-in and non-markovian points from the data.
        * Extract tex names.

    Note how MontePython takes the _folder_ to the chain files as root,
    different from the root for GetDist.

    Parameters
    ----------
    root: str
        Path to _folder_ containing the MontePython .txt chain files.
        E.g. for the following two chain files:
        /path/to/chains/Planck2015_TTlowP/2019-04-29_100000__1.txt
                                          2019-04-29_100000__2.txt
        you should pass the string "/path/to/chains/Planck2015_TTlowP".
        No trailing '/'.

    """
    if 'montepython' not in sys.modules:
        raise ImportError(
            "`montepython` not installed, but `MontePythonReader` needs "
            "the `montepython` package as an extra requirement in order "
            "to read the data folder. Alternatively you can pass the root "
            "to the actual chain _files_ into `MCMCSamples` (omitting "
            "_1.txt, _2.txt, etc.) in which case the files will be read "
            "by the `GetDistReader` (with less functionality, though).")
    # The variable and function names used here correspond to the ones
    # used in MontePython's analyze.py module.
    command_line = Namespace(files=[root])
    info = analyze.Information(command_line)
    analyze.prepare(files=command_line.files, info=info)
    analyze.extract_parameter_names(info=info)
    analyze.find_maximum_of_likelihood(info=info)
    data_per_chain = analyze.remove_bad_points(info=info)
    data = np.concatenate(data_per_chain, axis=0)

    params = info.ref_names
    tex = dict(zip(info.ref_names, info.tex_names))

    weights, minuslogL, data = np.split(data, [1, 2])

    columns = kwargs.pop('columns', params)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    return MCMCSamples(data=data, columns=columns,
                       weights=weights.flatten(), logL=-minuslogL, tex=tex,
                       root=root, *args, **kwargs)


class Namespace(object):
    """Input class for MontePython analyze module.

    Essentially none of this will be needed for anesthetic, however, in order
    to use MontePython's function `remove_bad_points` an instance of this
    class is required as input.
    """

    def __init__(self, files):
        self.bins = 20
        self.center_fisher = False
        self.contours_only = False
        self.decimal = 3
        self.extension = 'pdf'
        self.files = files
        self.fontsize = 16
        self.gaussian_smoothing = 0.5
        self.interpolation_smoothing = 4
        self.keep_fraction = 1.0
        self.legend_style = 'sides'
        self.line_width = 4
        self.markovian = True
        self.mean_likelihood = True
        self.minimal = False
        self.num_columns_1d = None
        self.only_markovian = False
        self.optional_plot_file = ''
        self.plot = False
        self.plot_2d = False
        self.plot_fisher = False
        self.posterior_smoothing = 5
        self.short_title_1d = False
        self.silent = False
        self.subparser_name = 'info'
        self.subplot = False
        self.temperature = 1.0
        self.ticknumber = 3
        self.ticksize = 14
        self.verbose = True
        self.want_covmat = False
