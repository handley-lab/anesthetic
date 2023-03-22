from anesthetic.scripts import gui
from bin.utils import unit_incremented
import pytest
import pandas._testing as tm


@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    tm.close()


def test_gui():
    gui(['./tests/example_data/pc', '--params', 'x0', 'x1', 'x2'])


@pytest.mark.parametrize('a, b', [['2.0.0b2', '2.0.0b1'],
                                  ['2.0.0b0', '2.0.0a3'],
                                  ['2.0.0', '2.0.0b1'],
                                  ['3.0.0a0', '2.5.6'],
                                  ['2.0.3', '2.0.2'],
                                  ['2.1.0', '2.0.5'],
                                  ['3.0.0', '2.5.6'],
                                  ])
def test_is_unit_incremented(a, b):
    assert unit_incremented(a, b)


@pytest.mark.parametrize('a, b', [['2.0.0b3', '2.0.0b1'],
                                  ['2.0.0b3', '2.0.0a3'],
                                  ['2.0.1', '2.0.0b1'],
                                  ['3.0.0a1', '2.5.6'],
                                  ['2.0.4', '2.0.2'],
                                  ['2.1.5', '2.0.5'],
                                  ['3.5.6', '2.5.6'],
                                  ])
def test_is_not_unit_incremented(a, b):
    assert not unit_incremented(a, b)
