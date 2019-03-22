import numpy

def read_paramnames(root):
    paramnames = []
    tex = {}
    paramnames_file = root + '.paramnames'
    with open(paramnames_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            paramname = line[0].replace('*', '')
            paramnames.append(paramname)
            tex[paramname] = ''.join(line[1:])
    return paramnames, tex

def read_ranges(root):
    prior_range_file = root + '.ranges'
    prior_range = {}
    try:
        with open(prior_range_file, 'r') as f:
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
                prior_range[paramname] = (xmin, xmax)
    except IOError:
        pass

    return prior_range

def read_birth(root):
    birth_file = root + '_dead-birth.txt'
    return numpy.loadtxt(birth_file)
