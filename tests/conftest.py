"""Make the package importable without an install (mirrors the prototype style)."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "examples"))   # tests cover the examples too
os.environ.setdefault("PTO_LIB_PATH", "/home/vloncar/work/einsum_workspace/pto-isa")

# The NPU test modules `import torch_npu`, which initializes the device — on the
# shared box that hangs or errors if a neighbour job has wedged the chip. So they are
# not even *collected* unless explicitly requested with PTO_RUN_NPU=1; the default
# `pytest` run is the device-free suite. (Set PTO_RUN_NPU=1 on a healthy box to add
# the device tests, then `-m "not npu"` still selects within them if needed.)
collect_ignore_glob = [] if os.environ.get("PTO_RUN_NPU") else ["*_npu.py"]


def healthy_npu():
    """A healthy, idle NPU device string (set as current), or None if none is usable.

    The box is shared: a chip pinned by a neighbour job, or wedged by an aicore
    timeout, is skipped so tests run on a usable device rather than flaking.
    Selection is driven by ``npu-smi info`` (Health OK + no running process) via
    ``common.device.pick_device`` — see that module for the parsing. If the driver
    does not answer promptly it raises ``NpuUnresponsive`` (hardware fault → stop and
    report), which we let propagate rather than silently skipping."""
    from common.device import pick_device  # examples/ is on sys.path (above)
    return pick_device()
