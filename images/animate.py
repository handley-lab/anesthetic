from matplotlib.animation import FuncAnimation
import numpy
from anesthetic import NestedSamples
nested = NestedSamples.read('plikHM_TTTEEE_lowl_lowE_lensing_NS/NS_plikHM_TTTEEE_lowl_lowE_lensing')
#nested = NestedSamples.read('./tests/example_data/ns/ns')

plotter = nested.gui(['omegam','H0','sigma8'])
plotter.param_choice.buttons.set_active(1)
plotter.param_choice.buttons.set_active(2)
plotter.fig.set_size_inches(5,6)
plotter.fig.tight_layout()

def update(i):
    print(i)
    plotter.evolution.slider.set_val(i)
    plotter.update(None)

frames = numpy.arange(plotter.evolution.slider.valmin, 50000, 2000)
anim = FuncAnimation(plotter.fig, update, frames=frames)
anim.save('images/anim.gif', writer='imagemagick')
