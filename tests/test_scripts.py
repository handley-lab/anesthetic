from anesthetic.scripts import gui
import matplotlib.pyplot as plt


def test_gui():
    gui('./tests/example_data/pc', '--params', 'x0', 'x1', 'x2')
    plt.close()
