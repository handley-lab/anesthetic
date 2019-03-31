import numpy

def read_paramnames(root):
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
                    tex[paramname] = '$' + ''.join(line[1:]) + '$'
        return paramnames, tex
    except IOError:
        return None, None

def read_limits(root):
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
    birth_file = root + '_dead-birth.txt'
    data = numpy.loadtxt(birth_file)
    params, logL, logL_birth  = numpy.split(data,[-2,-1], axis=1)
    return params, logL, logL_birth


def read_chains(root):
    chains_file = root + '_1.txt'
    data = numpy.loadtxt(chains_file)
    weights, chi2, params = numpy.split(data,[1,2], axis=1)
    logL = chi2/-2
    return weights, logL, params
