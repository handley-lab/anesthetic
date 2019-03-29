""" Tools for reading chains as produced by MultiNest and PolyChord. """
import numpy
import os


class NestedSamplingFiles(object):
    """ Class for accessing files produced by nested sampling.

    Args:
        root (str):
            root name of files.
            e.g. './chains/gaussian-'
    """
    def __init__(self, root):
        self.root = root

    @property
    def paramnames(self):
        """ Get the paramnames from <root>.paramnames. """
        with self.paramnames_file as f:
            return numpy.array([line.split()[0].replace('*', '')
                                for line in f])

    @property
    def latex(self):
        """ Get the latex symbols from <root>.paramnames. """
        with self.paramnames_file as f:
            return numpy.array([r'$' + ' '.join(line.split()[1:]) + r'$'
                                for line in f])

    @property
    def dead_file(self):
        """ Open the <root>dead-birth.txt file."""
        return self._open_file('dead-birth.txt')

    @property
    def phys_live_file(self):
        """ Open the <root>phys_live-birth.txt file."""
        return self._open_file('phys_live-birth.txt')

    @property
    def paramnames_file(self):
        """ Open the <root>.paramnames file."""
        return self._open_file('.paramnames')

    def _open_file(self, ext):
        """ Try opening files in MultiNest or PolyChord format.

        Ignores any '-' or '_' differences.

        Args:
            ext (str):
                File extension.
        """
        joins = ['', '_', '-']
        fnames = [self.root + s + ext for s in joins]
        for fname in fnames:
            if os.path.isfile(fname):
                return open(fname, 'r')


class NestedSamplingRun(object):
    """ Tool for extracting a full nested sampling run as a numpy array.

    Args:
        root (str):
            root name of files.
            e.g. './chains/gaussian-'
    """
    def __init__(self, root):
        self.files = NestedSamplingFiles(root)
        self.reload_data()

    def reload_data(self):
        """ Reload the data from the files. """

        dead_data = numpy.loadtxt(self.files.dead_file)
        phys_live_data = numpy.loadtxt(self.files.phys_live_file)
        nparams = len(self.files.paramnames)

        self._data = numpy.concatenate((dead_data[:, :nparams+2],
                                        phys_live_data[:, :nparams+2]))
        self._data = numpy.unique(self._data, axis=0)
        self._data = self._data[self._data[:, nparams].argsort(), :]

        # Check this actually makes sense...
        self.n = numpy.array([numpy.count_nonzero(self._indices(logL))
                              for logL in
                              [self.logL0[0]] + list(self.logL[:-1])])
        self.logw = -1./self.n
        self.logX = numpy.cumsum(self.logw)
        self.logw += self.logX
        LX = 0.5*self.logL + self.logX
        LX -= numpy.max(LX)
        LX = numpy.exp(LX)

    def live_points(self, logL):
        """ Physical coordinates of live points strictly above logL. """
        return self.points[self._indices(logL), :]

    def live_likes(self, logL):
        """ Likelihoods of the live points strictly above logL. """
        return self.logL[self._indices(logL)]

    def _indices(self, logL):
        """ Indices in the _data array of live points strictly above logL. """
        return numpy.logical_and(self.logL > logL, self.logL0 <= logL)

    @property
    def npoints(self):
        """ Number of dead points. """
        return self._data.shape[0]

    @property
    def points(self):
        """ Dead points. """
        return self._data[:, :-2]

    @property
    def logL(self):
        """ Death contours. """
        return self._data[:, -2]

    @property
    def logL0(self):
        """ Birth contours. """
        return self._data[:, -1]

    def LX(self, kT=1):
        """ Likelihood * volume compression, at a given temperature
        (normalised to 1) """
        logLX = self.logL/kT + self.logX
        logLX -= numpy.max(logLX)
        return numpy.exp(logLX)

    def posterior_points(self, kT=1):
        """ Equally weighted posterior samples at a given temperature """
        numpy.random.seed(int(kT*10000))
        indices = numpy.random.rand(len(self.LX(kT))) < self.LX(kT)
        return self.points[indices, :]
