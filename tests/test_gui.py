import anesthetic.examples._matplotlib_agg  # noqa: F401
from anesthetic import read_chains
import pytest
import pandas._testing as tm


@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    tm.close()


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

    # Reload button
    plotter.reload.button.on_clicked(plotter.reload_file(None))

    # Reset button
    plotter.reset.button.on_clicked(plotter.reset_range(None))


def test_gui_params():
    plotter = read_chains('./tests/example_data/pc').gui()
    assert len(plotter.param_choice.buttons.labels) == 8

    plotter = read_chains('./tests/example_data/pc').gui(params=['x0', 'x1'])
    assert len(plotter.param_choice.buttons.labels) == 2


def test_slider_reset_range():
    plotter = read_chains('./tests/example_data/pc').gui()
    plotter.evolution.reset_range(-3, 3)
    assert plotter.evolution.ax.get_xlim() == (-3.0, 3.0)
