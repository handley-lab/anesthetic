import argparse
import matplotlib.pyplot as plt
from anesthetic import read_chains

def gui():
    description = "Nested sampling visualisation"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('file_root', type=str,
                        help="File name root of nested sampling run")
    parser.add_argument('--params', '-p', nargs='*', type=str,
                        help="Parameters to display")
    args = parser.parse_args()

    samples = read_chains(root=args.file_root)
    plotter = samples.gui(args.params)
    plt.show()
