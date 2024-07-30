"""Read NestedSamples from dnest4 output files"""
import os
import numpy as np
from anesthetic.samples import DiffusiveNestedSamples


def heuristically_determine_columns(n_params):
    """
    heuristically determines column names. If none are given, parameters are named x_i
    """
    ## TODO: check if user gave DNest4 a list of parameter names for the model
    return [f'x{i}' for i in range(n_params)]


def read_dnest4(root,
                *args,
                **kwargs):
    """Read dnest4 output files.

    Parameters
    ----------
    levels_file default output name from dnest4
    sample_file default output name from dnest4
    sample_info_file default output name from dnest4
    root : str
        root specify the directory only, no specific roots,
        The files read files are levels_file, sample_file and sample_info.
    """
    levels_file = 'levels.txt'
    sample_file = 'sample.txt'
    sample_info_file = 'sample_info.txt'
    weights_file = 'weights.txt'
    prior_weights_file = 'log_prior_weights.txt'
    posterior_weights = 'p_samples0.txt'
    logx_samples = 'logx_samples0.txt'

    print(os.path.join(root, levels_file))

    levels = np.loadtxt(os.path.join(root, levels_file), dtype=float, delimiter=' ', comments='#')
    samples = np.genfromtxt(os.path.join(root, sample_file), dtype=float, delimiter=' ', comments='#', skip_header=1)
    sample_info = np.loadtxt(os.path.join(root, sample_info_file), dtype=float, delimiter=' ', comments='#')
    weights = np.loadtxt(os.path.join(root, weights_file), dtype=float, delimiter=' ', comments='#')
    prior_weights = np.loadtxt(os.path.join(root, prior_weights_file), dtype=float, delimiter=' ', comments='#')
    logx_samples = np.genfromtxt(os.path.join(root, logx_samples), dtype=float, delimiter=' ', comments='#', skip_header=0)
    posterior_weights = np.genfromtxt(os.path.join(root, posterior_weights), dtype=float, delimiter=' ', comments='#', skip_header=0)
    n_params = samples.shape[1]
    columns = heuristically_determine_columns(n_params)

    return DiffusiveNestedSamples(samples=samples,
                      sample_info=sample_info,
                      levels=levels,
                      columns=columns,
                      weights=weights,
                      prior_weights=prior_weights,
                      logx_samples=logx_samples,
                      posterior_weights=posterior_weights,
                      logL=sample_info[:, 1],
                      labels=columns,
                      *args,
                      **kwargs)
