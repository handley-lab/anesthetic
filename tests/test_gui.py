from anesthetic.gui import RunPlotter


def test_gui():
    plotter = RunPlotter('./tests/example_data/ns/ns')

    # Type buttons
    plotter.type.buttons.set_active(0)
    assert(plotter.type() == 'live')
    plotter.type.buttons.set_active(1)
    assert(plotter.type() == 'posterior')

    # Parameter choice buttons
    plotter.param_choice.buttons.set_active(1)
    assert(len(plotter.triangle.ax) == 2)
    plotter.param_choice.buttons.set_active(0)
    assert(len(plotter.triangle.ax) == 1)
    plotter.param_choice.buttons.set_active(0)
    plotter.param_choice.buttons.set_active(2)
    plotter.param_choice.buttons.set_active(3)
    assert(len(plotter.triangle.ax) == 4)
