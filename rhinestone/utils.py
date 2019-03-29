"""General utilities."""


def min_max(x):
    """ Get the minimum and maximum of x at the same time.

    Args:
        x (array-like):
            anything that can be acted on by min() and max() functions.

    Returns:
        tuple:
            min and max of x
    """
    return min(x), max(x)
