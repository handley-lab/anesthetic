"""Main plotting tools."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import (GridSpec as GS,
                                 GridSpecFromSubplotSpec as sGS)
from anesthetic.gui.widgets import (Widget, Slider, Button,
                                    RadioButtons, TrianglePlot, CheckButtons)


class Higson(Widget):
    """Higson plot as shown in https://arxiv.org/abs/1703.09701 .

    Attributes
    ----------
        curve : :class:`matplotlib.lines.Line2D`
            points currently plotted as a curve.

        point : :class:`matplotlib.lines.Line2D`
            large indicator point currently plotted on the curve.

    """

    def __init__(self, fig, gridspec):
        super().__init__(fig, gridspec)
        self.ax.set_yticks([])
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_ylabel(r'$LX$')
        self.ax.set_xlabel(r'$\ln X$')

        self.curve, = self.ax.plot([None], [None], 'k-')
        self.point, = self.ax.plot([None], [None], 'ko')

    def update(self, logX, LX, i):
        """Update the line and the point in the higson plot.

        Parameters
        ----------
            logX : array-like
                log-volume compression values to plot

            LX : array-like
                Likelihood * volume compression

            i : int
                Current location of higson point

        """
        self.point.set_xdata(logX[[i]])
        self.point.set_ydata(LX[[i]])
        self.curve.set_xdata(logX)
        self.curve.set_ydata(LX)

    def reset_range(self):
        """Reset the ranges of the higson plot."""
        xdata = self.curve.get_xdata()
        xdata = xdata[np.isfinite(xdata)]
        self.ax.set_xlim(xdata.max(), xdata.min())


class Evolution(Slider):
    """Slider controlling the evolution stage of the live points."""

    def __init__(self, logX, fig, gridspec, action):
        self._logX = logX
        super().__init__(fig, gridspec, action, '',
                         -logX[0], -logX[-1], -logX[0], 'horizontal')
        self.slider.valtext.set_horizontalalignment('right')
        self.slider.valtext.set_position((0.98, 0.5))

    def __call__(self):
        """Return the current iteration as an integer."""
        return np.argmin(super().__call__() > -self._logX)

    def set_text(self, logL, n):
        """Set the text at end of slider.

        Parameters
        ----------
            logL : float
                Current loglikelihood of evolution stage

            n : int
                Current number of live points of evolution stage

        """
        text = r'$\ln L$: %.6g, $n_\mathrm{live}$: %i' % (logL, n)
        return super().set_text(text)


class Beta(Slider):
    """Slider controlling inverse temperature of the posterior points."""

    def __init__(self, beta, D_KL, fig, gridspec, action):
        self._beta = beta
        self._D_KL = D_KL
        D_KL0 = np.interp(1, beta, D_KL)
        super().__init__(fig, gridspec, action, r'$\beta$',
                         D_KL[0], D_KL[-1], D_KL0, 'vertical')

    def __call__(self):
        """Return the current inverse temperature."""
        return np.interp(super().__call__(), self._D_KL, self._beta)

    def set_text(self, beta):
        """Set the text at end of slider.

        Parameters
        ----------
            beta : float
                Current inverse temperature of posterior points stage

        """
        text = r'%.2g' % beta
        return super().set_text(text)


class RunPlotter(object):
    """Construct a control panel of information on a nested sampling run.

    Parameters
    ----------
    samples : :class:`anesthetic.samples.NestedSamples`
        The root string for the chains files to be used, or a set of nested
        samples.

    Attributes
    ----------
    samples : :class:`anesthetic.samples.NestedSamples`
        Object for extracting nested sampling data from chains files.

    fig : :class:`matplotlib.figure.Figure`
        Reference to the underlying figure

    triangle : :class:`anesthetic.gui.widgets.TrianglePlot`
        Corner plot of live or posterior samples.

    beta : :class:`anesthetic.gui.plot.Beta`
        Slider selecting the posterior inverse temperature.

    evolution : :class:`anesthetic.gui.plot.Evolution`
        Slider selecting the live iteration.

    higson : :class:`anesthetic.gui.plot.Higson`
        Higson plot of posterior weights.

    reset : :class:`anesthetic.gui.widgets.Button`
        Button that resets the parameter ranges.

    reload : :class:`anesthetic.gui.widgets.Button`
        Button that reloads the files.

    type : :class:`anesthetic.gui.widgets.RadioButtons`
        Radio buttons that selects whether to plot live or posteriors.

    param_choice : :class:`anesthetic.gui.widgets.CheckButtons`
        Checkbox that selects which parameters to plot.

    """

    def __init__(self, samples, params=None):
        self.samples = samples

        if params:
            self.params = np.array(params)
        else:
            self.params = np.array(self.samples.drop_labels().columns[:10])

        self.fig = plt.figure()
        self._set_up()
        self.redraw(None)

    def _set_up(self):
        """Draw the control panel.

        We implement the control panel using sequential recursive gridspecs.

        +----------------------------------------------+------+
        |                                              |      |
        |                  gs0[0]                      |gs0[1]|
        |                                              |      |
        |                                              |      |
        |                                              |      |
        |                                              |      |
        |                                              |      |
        |                                              |      |
        +-------------------------+-----------------+--+------+
        |                         |    gs11[0]      |         |
        |         gs10[0]         +-----------------+  gs1[2] |
        +-------------------------+    gs11[1]      |         |
        |                         +-----------------+         |
        |         gs10[1]         |    gs11[2]      |         |
        +-------------------------+-----------------+---------+

        These variable names are included in the __init__ function, and are
        named with an intuitive Huffman coding.

        """
        gs = GS(2, 1, height_ratios=[3, 1])
        gs0 = sGS(1, 2, width_ratios=[19, 1], subplot_spec=gs[0])
        gs1 = sGS(1, 3, width_ratios=[4, 1, 1], subplot_spec=gs[1])
        gs10 = sGS(2, 1, height_ratios=[1, 4], subplot_spec=gs1[0])
        gs11 = sGS(3, 1, height_ratios=[1, 1, 2], subplot_spec=gs1[1])

        self.triangle = TrianglePlot(self.fig, gs0[0])
        beta = np.logspace(-10, 10, 101)
        D_KL = self.samples.D_KL(beta=beta).to_numpy()
        self.beta = Beta(beta, D_KL, self.fig, gs0[1], self.update)
        logX = self.samples.logX().to_numpy()
        self.evolution = Evolution(logX, self.fig, gs10[0], self.update)
        self.higson = Higson(self.fig, gs10[1])
        self.reset = Button(self.fig, gs11[0],
                            self.reset_range, 'Reset Range')
        self.reload = Button(self.fig, gs11[1],
                             self.reload_file, 'Reload File')
        self.type = RadioButtons(self.fig, gs11[2],
                                 ('live', 'posterior'), self.update)
        self.param_choice = CheckButtons(self.fig, gs1[2],
                                         self.params, self.redraw)

    def redraw(self, _):
        """Redraw the triangle plot upon parameter updating."""
        self.triangle.draw(self.param_choice(),
                           self.samples.get_labels_map())
        self.update(None)
        self.reset_range(None)
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def points(self, label):
        """Get sample coordinates from nested sampling samples.

        Parameters
        ----------
        label : str
            label indicating the coordinate to extract.

        Returns
        -------
        array-like:
            sample 'label'-coordinates.

        """
        if self.type() == 'posterior':
            beta = self.beta()
            return self.samples.posterior_points(beta)[label]
        else:
            i = self.evolution()
            logL = self.samples.logL.iloc[i]
            return self.samples.live_points(logL)[label]

    def update(self, _):
        """Update all the plots upon slider changes."""
        logX = np.log(self.samples.nlive / (self.samples.nlive+1)).cumsum()
        beta = self.beta()
        LX = self.samples.logL*beta + logX
        LX = np.exp(LX-LX.max())
        i = self.evolution()
        logL = self.samples.logL.iloc[i]
        n = self.samples.nlive.iloc[i]

        self.triangle.update(self.points)

        self.evolution.set_text(logL, n)
        self.beta.set_text(beta)

        self.higson.update(logX, LX, i)
        self.fig.canvas.draw()

    def reload_file(self, _):
        """Reload the data from file."""
        from anesthetic import read_chains
        self.samples = read_chains(self.samples.root)
        self.evolution.reset_range(valmax=len(self.samples))
        self.update(None)

    def reset_range(self, _):
        """Reset the parameter ranges."""
        self.triangle.reset_range()
        self.higson.reset_range()
        self.fig.canvas.draw()
