"""Widget wrappers to matplotlib.

These extend the matplotlib widgets by plotting themselves onto an axis and
storing a reference to both the widget object and the axis on which they are
plotted.

"""

from matplotlib.widgets import Button as mplButton
from matplotlib.widgets import CheckButtons as mplCheckButtons
from matplotlib.widgets import RadioButtons as mplRadioButtons
from anesthetic.gui._matplotlib import Slider as mplSlider
from anesthetic.utils import histogram
from anesthetic.plot import make_2d_axes


class Widget(object):
    """Base class for anesthetic gui widgets.

    Stores a reference to the underlying figure, the gridspec that the
    widget is placed at and the axes of the widget.

    Parameters
    ----------
        fig: matplotlib.figure.Figure
            Figure for drawing widget on.

        gridspec: matplotlib.gridspec.GridSpec
            Specification for where to draw in the figure.
            Technically could be any argument that can be passed to
            matplotlib.figure.Figure.add_subplot.

    Attributes
    ----------
        fig: (matplotlib.figure.Figure
            Figure for drawing widget on.

        gridspec: matplotlib.gridspec.GridSpec
            Specification for where to draw in the figure.
            Technically could be any argument that can be passed to
            matplotlib.figure.Figure.add_subplot.

        ax: matplotlib.axes.Axes
            Axes of widget.

    """

    def __init__(self, fig, gridspec):
        self.fig = fig
        self.gridspec = gridspec
        self.ax = self.fig.add_subplot(self.gridspec)


class LabelsWidget(Widget):
    """Widget with labels to choose from.

    Parameters
    ----------
        labels: list(str)
            Set of labels to be tied to Widget.

    Attributes
    ----------
        labels: list(str)
            Set of labels on Widget.

    """

    def __init__(self, fig, gridspec, labels):
        super(LabelsWidget, self).__init__(fig, gridspec)
        self.labels = labels


class Button(Widget):
    """Push button that performs an action.

    Parameters
    ----------
        action: func
            What should be run upon clicking the button.

        text: str
            Overlay text on the button.

    Attributes
    ----------
        button: matplotlib.widgets.Button
            matplotlib Button.

    """

    def __init__(self, fig, gridspec, action, text):
        super(Button, self).__init__(fig, gridspec)
        self.button = mplButton(self.ax, text)
        self.button.on_clicked(action)


class CheckButtons(LabelsWidget):
    """Set of checkboxes.

    Defaults to choosing the only the first label at start.

    Parameters
    ----------
        action: func
            What should be done upon checking the box.

    Attributes
    ----------
        buttons: matplotlib.widgets.CheckButtons
            matplotlib CheckButtons.

    """

    def __init__(self, fig, gridspec, labels, action):
        super(CheckButtons, self).__init__(fig, gridspec, labels)
        default = [i == 0 for i, _ in enumerate(self.labels)]
        self.buttons = mplCheckButtons(self.ax, self.labels, default)
        self.buttons.on_clicked(action)

    def __call__(self):
        """Get current status of the check boxes."""
        return self.labels[list(self.buttons.get_status())]


class RadioButtons(LabelsWidget):
    """Set of radio selection choices.

    Parameters
    ----------
        labels: list(str))

    Attributes
    ----------
        buttons: matplotlib.widgets.RadioButtons
            matplotlib RadioButtons.

    """

    def __init__(self, fig, gridspec, labels, action):
        super(RadioButtons, self).__init__(fig, gridspec, labels)
        self.buttons = mplRadioButtons(self.ax, self.labels)
        self.buttons.on_clicked(action)

    def __call__(self):
        """Get current selection string."""
        return self.buttons.value_selected


class Slider(Widget):
    """Choose a parameter via a slider widget.

    Parameters
    ----------
        action: func
            What should be done upon altering the slider.

        text: str
            Text at the start of the slider.

        valmin, valmax, valinit: float
            Range and initial value for the slider.

        orientation: str
            Orientation for slider ('horizontal' or 'vertical').

    Attributes
    ----------
        slider: matplotlib.widgets.Slider
            matplotlib Slider.

    """

    def __init__(self, fig, gridspec, action, text,
                 valmin, valmax, valinit, orientation):
        super(Slider, self).__init__(fig, gridspec)
        self.slider = mplSlider(self.ax, text, valmin, valmax, valinit=valinit,
                                orientation=orientation)
        self.slider.on_changed(action)

    def set_text(self, text):
        """Set the text at the end of the slider.

        Parameters
        ----------
            text: str
                Text to be chosen

        """
        self.slider.valtext.set_text(text)

    def reset_range(self, valmin=None, valmax=None):
        """Reset the range of the slider.

        Kwargs:
            valmin, valmax: (float), optional:
                The new limits to the slider.

        """
        if valmax is not None:
            self.slider.valmax = valmax
        if valmin is not None:
            self.slider.valmin = valmin
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)

    def __call__(self):
        """Return the float value in the slider."""
        return self.slider.val


class TrianglePlot(Widget):
    """Triangle plot widget as exemplified by getdist and corner.py.

    For other examples of these plots, see:
    https://getdist.readthedocs.io
    https://corner.readthedocs.io

    Attributes
    ----------
        ax: pandas.DataFrame(matplotlib.axes.Axes)
            Mapping from pairs of parameters to axes for plotting.
    """

    def __init__(self, fig, gridspec):
        super(TrianglePlot, self).__init__(fig, gridspec)
        self.fig.delaxes(self.ax)
        _, self.ax = make_2d_axes([], fig=self.fig, subplot_spec=self.gridspec)

    def draw(self, labels, tex={}):
        """Draw a new triangular grid for list of parameters labels.

        Parameters
        ----------
            labels: list(str)
                labels for the triangular grid.

        """
        # Remove any existing axes
        for y, row in self.ax.iterrows():
            for x, ax in row.iteritems():
                if ax is not None:
                    if x == y:
                        self.fig.delaxes(ax.twin)
                    self.fig.delaxes(ax)

        # Set up the axes
        _, self.ax = make_2d_axes(labels, upper=False, tex=tex,
                                  fig=self.fig, subplot_spec=self.gridspec)

        # Plot no points  points.
        for y, row in self.ax.iterrows():
            for x, ax in row.iteritems():
                if ax is not None:
                    if x == y:
                        ax.twin.plot([None], [None], 'k-')
                    else:
                        ax.plot([None], [None], 'k.')

    def update(self, f):
        """Update the points in the triangle plot using f function.

        Parameters
        ----------
            f: func: str -> array(float)
                this function should take in a parameter label i, and return an
                array-like object of the i-coordinate of the samples

        """
        for y, row in self.ax.iterrows():
            for x, ax in row.iteritems():
                if ax is not None:
                    if x == y:
                        datx, daty = histogram(f(x), bins='auto')
                        ax.twin.lines[0].set_xdata(datx)
                        ax.twin.lines[0].set_ydata(daty)
                    else:
                        ax.lines[0].set_xdata(f(x))
                        ax.lines[0].set_ydata(f(y))

    def reset_range(self):
        """Reset the range of each grid."""
        for y, row in self.ax.iterrows():
            for x, ax in row.iteritems():
                if ax is not None:
                    if x == y:
                        ax.twin.relim()
                        ax.twin.autoscale_view()
                    ax.relim()
                    ax.autoscale_view()
