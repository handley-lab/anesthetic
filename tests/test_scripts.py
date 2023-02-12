from anesthetic.scripts import gui
import pytest
import pandas._testing as tm


@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    tm.close()


def test_gui():
    gui('./tests/example_data/pc', '--params', 'x0', 'x1', 'x2')
