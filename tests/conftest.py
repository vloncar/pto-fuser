"""Make the package importable without an install (mirrors the prototype style)."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.environ.setdefault("PTO_LIB_PATH", "/home/vloncar/work/einsum_workspace/pto-isa")


def healthy_npu():
    """First NPU that passes a tiny matmul, set as current device. The box is
    shared: a chip pinned by a neighbor job (or wedged by an aicore timeout) is
    skipped so tests run on a usable device rather than flaking. Returns the device
    string, or None if no NPU is healthy (caller skips)."""
    import torch
    if not torch.npu.is_available():
        return None
    for i in range(torch.npu.device_count()):
        d = f"npu:{i}"
        try:
            torch.npu.set_device(d)
            x = torch.randn(64, 64, device=d).half()
            _ = x @ x
            torch.npu.synchronize()
            return d
        except Exception:
            continue
    return None
