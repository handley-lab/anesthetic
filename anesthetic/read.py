"""Tools for reading from chains files.

This acts as an interface to MCMC samplers. Currently provide reading for

- GetDist
- PolyChord

More will be added in future
"""
import numpy


def read_paramnames(root):
    r"""Read ``<root>.paramnames`` in getdist format.

    This file should contain one or two columns. The first column indicates a
    reference name for the sample, used as labels in the pandas array. The
    second optional column should include the equivalent axis label, possibly
    in tex, with the understanding that it will be surrounded by dollar signs,
    for example

    <root.paramnames>

    a1     a_1
    a2     a_2
    omega  \omega
    """
    try:
        paramnames = []
        tex = {}
        paramnames_file = root + '.paramnames'
        with open(paramnames_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                paramname = line[0].replace('*', '')
                paramnames.append(paramname)
                if len(line) > 1:
                    tex[paramname] = '$' + ' '.join(line[1:]) + '$'
        return paramnames, tex
    except IOError:
        return None, None


def read_limits(root):
    """Read ``<root>.ranges`` in getdist format."""
    limits_file = root + '.ranges'
    limits = {}
    try:
        with open(limits_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                paramname = line[0]
                try:
                    xmin = float(line[1])
                except ValueError:
                    xmin = None
                try:
                    xmax = float(line[2])
                except ValueError:
                    xmax = None
                limits[paramname] = (xmin, xmax)
    except IOError:
        pass

    return limits


def read_birth(root):
    """Read ``<root>_dead-birth.txt`` in polychord format."""
    birth_file = root + '_dead-birth.txt'
    data = numpy.loadtxt(birth_file)
    samples, logL, logL_birth = numpy.split(data, [-2, -1], axis=1)
    return samples, logL, logL_birth


def read_chains(root):
    """Read ``<root>_1.txt`` in getdist format."""
    chains_file = root + '_1.txt'
    data = numpy.loadtxt(chains_file)
    weights, chi2, samples = numpy.split(data, [1, 2], axis=1)
    logL = chi2/-2
    return weights, logL, samples
