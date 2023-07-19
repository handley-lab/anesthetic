import sys
import anesthetic.examples._matplotlib_agg  # noqa: F401
from anesthetic import read_chains
import pytest
import pandas._testing as tm
try:
    import h5py  # noqa: F401
except ImportError:
    pass


@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    tm.close()


@pytest.mark.parametrize('root', ["./tests/example_data/pc",
                                  "./tests/example_data/mn",
                                  "./tests/example_data/un"])
def test_gui(root):
    if 'un' in root and 'h5py' not in sys.modules:
        pytest.skip("`h5py` not in sys.modules, but needed for ultranest.")
    samples = read_chains(root)
    plotter = samples.gui()

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
    old = plotter.evolution()
    plotter.evolution.slider.set_val(5)
    assert plotter.evolution() != old
    plotter.evolution.slider.set_val(0)
    assert plotter.evolution() == old
    plotter.type.buttons.set_active(1)

    plotter.beta.slider.set_val(0)
    assert plotter.beta() == pytest.approx(0, 0, 1e-8)

    plotter.beta.slider.set_val(samples.D_KL())
    assert plotter.beta() == pytest.approx(1)
    plotter.beta.slider.set_val(1e2)
    assert plotter.beta() == 1e10
    plotter.type.buttons.set_active(0)

    # Reload button
    plotter.reload.button.on_clicked(plotter.reload_file(None))

    # Reset button
    plotter.reset.button.on_clicked(plotter.reset_range(None))


@pytest.mark.parametrize('root', ["./tests/example_data/pc",
                                  "./tests/example_data/mn",
                                  "./tests/example_data/un"])
def test_gui_params(root):
    if 'un' in root and 'h5py' not in sys.modules:
        pytest.skip("`h5py` not in sys.modules, but needed for ultranest.")
    samples = read_chains(root)
    params = samples.columns.get_level_values(0).to_list()
    plotter = samples.gui()
    assert len(plotter.param_choice.buttons.labels) == len(params)

    plotter = samples.gui(params=params[:2])
    assert len(plotter.param_choice.buttons.labels) == 2


@pytest.mark.parametrize('root', ["./tests/example_data/pc",
                                  "./tests/example_data/mn",
                                  "./tests/example_data/un"])
def test_slider_reset_range(root):
    if 'un' in root and 'h5py' not in sys.modules:
        pytest.skip("`h5py` not in sys.modules, but needed for ultranest.")
    plotter = read_chains(root).gui()
    plotter.evolution.reset_range(-3, 3)
    assert plotter.evolution.ax.get_xlim() == (-3.0, 3.0)
