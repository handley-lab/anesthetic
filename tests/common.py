import pytest
import pandas._testing as tm

@pytest.fixture(autouse=True)
def close_figures_on_teardown():
    yield
    tm.close()
