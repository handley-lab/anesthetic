import matplotlib_agg  # noqa: F401
from packaging import version
from matplotlib import __version__ as mpl_version
from anesthetic import read_chains


def test_gui():
    plotter = read_chains('./tests/example_data/pc').gui()

    # Type buttons
    plotter.type.buttons.set_active(1)
    assert plotter.type() == 'posterior'
    plotter.type.buttons.set_active(0)
    assert plotter.type() == 'live'

    # Parameter choice buttons
    plotter.param_choice.buttons.set_active(1)
    assert len(plotter.triangle.ax) == 2
    plotter.param_choice.buttons.set_active(0)
    assert len(plotter.triangle.ax) == 1
    plotter.param_choice.buttons.set_active(0)
    plotter.param_choice.buttons.set_active(2)
    plotter.param_choice.buttons.set_active(3)
    assert len(plotter.triangle.ax) == 4

    # Sliders
    plotter.evolution.slider.set_val(100)
    assert plotter.evolution() == 100
    plotter.type.buttons.set_active(1)

    plotter.temperature.slider.set_val(0)
    assert plotter.temperature() == 1
    plotter.temperature.slider.set_val(1)
    assert plotter.temperature() == 10
    plotter.temperature.slider.set_val(2)
    assert plotter.temperature() == 100
    plotter.type.buttons.set_active(0)

    if version.parse(mpl_version) >= version.parse('3.6.0'):
        pass
        # TODO: observers have been removed in mpl 3.6
    elif version.parse(mpl_version) >= version.parse('3.4.0'):
        # Reload button
        plotter.reload.button.observers[0]()

        # Reset button
        plotter.reset.button.observers[0]()
    else:
        # Reload button
        plotter.reload.button.observers[0](None)

        # Reset button
        plotter.reset.button.observers[0](None)


def test_gui_params():
    plotter = read_chains('./tests/example_data/pc').gui()
    assert len(plotter.param_choice.buttons.labels) == 8

    plotter = read_chains('./tests/example_data/pc').gui(params=['x0', 'x1'])
    assert len(plotter.param_choice.buttons.labels) == 2


def test_slider_reset_range():
    plotter = read_chains('./tests/example_data/pc').gui()
    plotter.evolution.reset_range(-3, 3)
    assert plotter.evolution.ax.get_xlim() == (-3.0, 3.0)
