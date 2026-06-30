"""Shared benchmark + plotting infrastructure for the pto-fuser examples.

Importing this package wires ``src/`` onto ``sys.path`` (so the examples run from a
checkout without an install) and exposes the small set of helpers the examples
share: device selection, wall-clock timing, table formatting, and plotting.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.environ.setdefault("PTO_LIB_PATH", "/home/vloncar/work/einsum_workspace/pto-isa")

from .device import on_npu, pick_device                       # noqa: E402
from .bench import Measurement, format_table, time_ms         # noqa: E402
from .plot import plot_speedups, plot_sweep                   # noqa: E402

__all__ = ["on_npu", "pick_device", "Measurement", "format_table", "time_ms",
           "plot_speedups", "plot_sweep"]
