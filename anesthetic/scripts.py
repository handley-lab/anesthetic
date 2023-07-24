"""Command-line scripts for anesthetic."""
import argparse
import matplotlib.pyplot as plt
from anesthetic import read_chains


def gui(args=None):
    """Launch the anesthetic GUI.

    See :class:`anesthetic.gui.plot.RunPlotter` for details.
    """
    description = "Nested sampling visualisation"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('file_root', type=str,
                        help="File name root of nested sampling run")
    parser.add_argument('--params', '-p', nargs='*', type=str,
                        help="Parameters to display")
    args = parser.parse_args(args)

    samples = read_chains(root=args.file_root)
    _ = samples.gui(args.params)
    plt.show()
