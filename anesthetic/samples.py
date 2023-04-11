"""Deprecated module for backwards compatibility."""
# TODO: remove this file in version >= 2.1
import warnings
from anesthetic.core import *  # noqa: F403, F401

warnings.warn(
    "You are using the anesthetic.samples module, which is deprecated. "
    "Please use anesthetic.core instead.",
    FutureWarning
    )
