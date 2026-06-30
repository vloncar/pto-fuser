"""Device selection for the examples.

The dev box is shared and occasionally flaky: a chip pinned by a neighbour job, or
wedged by an aicore timeout, will fail (or *hang*) even on a known-good kernel. A blind
``x @ x`` probe on such a chip is exactly what we must avoid, so before touching the
device we ask the driver which chips are healthy and idle via ``npu-smi info``.

``npu-smi info`` prints two tables::

    | NPU   Name        | Health   | ...          <- one row per chip
    | Chip              | Bus-Id   | ...
    ...
    | No running processes found in NPU 2         <- process table (idle chips)
    | 3   0   | 12345 | python | 1234 |           <- or a process row (busy chip)

The leading number on each device row (``2``, ``3`` here) is the *physical* id (PCIe
slot). Torch addresses chips by *logical* id counting from 0 in the order they are
listed, so physical ``2, 3`` -> logical ``0, 1``. We pick the first chip whose Health is
``OK`` and which has no running process, and return its logical id.

If ``npu-smi info`` takes longer than a few seconds to answer, the chips have usually
suffered a fault that only a reboot clears — there is no point probing further. We raise
:class:`NpuUnresponsive` so the caller stops and reports rather than hanging or drawing a
wrong conclusion.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional

#: If ``npu-smi info`` is slower than this, treat the chips as wedged (reboot needed).
SMI_TIMEOUT_S = 3.0

_DEVICE_ROW = re.compile(r"^\|\s*(\d+)\s+\S+\s*\|\s*(OK|Warning|Alarm|Critical)\b")
_IDLE_ROW = re.compile(r"No running processes found in NPU\s+(\d+)")


class NpuUnresponsive(RuntimeError):
    """``npu-smi info`` did not answer promptly — the chips are likely wedged and need a
    reboot. Stop and report; do not keep probing."""


@dataclass(frozen=True)
class NpuStatus:
    logical: int   # torch device index (npu:<logical>), 0-based in listing order
    physical: int  # physical slot id reported by npu-smi
    health: str    # "OK" / "Warning" / "Alarm" / "Critical"
    free: bool      # True iff no process is running on the chip

    @property
    def usable(self) -> bool:
        return self.health == "OK" and self.free


def query_npus(timeout: float = SMI_TIMEOUT_S) -> List[NpuStatus]:
    """Parse ``npu-smi info`` into per-chip status, logical id assigned by listing order.

    Returns ``[]`` if ``npu-smi`` is not installed (i.e. not on an Ascend box). Raises
    :class:`NpuUnresponsive` if the command times out or errors — that is the signal to
    stop everything and report a hardware fault.
    """
    try:
        proc = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return []  # not an Ascend host
    except subprocess.TimeoutExpired as e:
        raise NpuUnresponsive(
            f"`npu-smi info` did not return within {timeout:g}s — the NPUs are likely "
            f"wedged and need a reboot. Stop and report; do not keep probing."
        ) from e
    if proc.returncode != 0:
        raise NpuUnresponsive(
            f"`npu-smi info` exited {proc.returncode} — driver fault, the NPUs are "
            f"likely unusable until a reboot. Stop and report.\n{proc.stderr.strip()}"
        )

    healths: List[tuple[int, str]] = []   # (physical, health) in listing order
    idle: set[int] = set()                # physical ids with no running process
    for line in proc.stdout.splitlines():
        m = _DEVICE_ROW.match(line)
        if m:
            healths.append((int(m.group(1)), m.group(2)))
            continue
        m = _IDLE_ROW.search(line)
        if m:
            idle.add(int(m.group(1)))

    return [
        NpuStatus(logical=i, physical=phys, health=health, free=phys in idle)
        for i, (phys, health) in enumerate(healths)
    ]


def free_npu(timeout: float = SMI_TIMEOUT_S) -> Optional[int]:
    """Logical id of the first ``OK`` + idle chip per ``npu-smi info``, else ``None``.

    Raises :class:`NpuUnresponsive` if the driver does not answer promptly.
    """
    for s in query_npus(timeout):
        if s.usable:
            return s.logical
    return None


def pick_device(prefer: Optional[str] = None) -> Optional[str]:
    """Return a healthy, idle NPU device string (set as current), or ``None`` off-NPU.

    Selection is driven by ``npu-smi info`` (Health ``OK`` + no running process) so we
    never touch a chip a neighbour job has pinned or wedged. ``prefer`` (e.g. ``npu:1``)
    is honoured only if that chip is reported usable. Raises :class:`NpuUnresponsive` if
    the driver does not answer — propagate it; that means stop and report a fault.
    """
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError:
        return None
    if not torch.npu.is_available():
        return None

    status = query_npus()
    if status:
        usable = [s.logical for s in status if s.usable]
        if prefer is not None:
            try:
                want = int(prefer.split(":")[1])
            except (IndexError, ValueError):
                want = None
            if want is not None and want in usable:
                usable = [want] + [u for u in usable if u != want]
        for logical in usable:
            d = f"npu:{logical}"
            try:
                torch.npu.set_device(d)
                return d
            except Exception:
                continue
        return None

    # npu-smi unavailable but torch sees NPUs: fall back to a tiny-matmul probe.
    candidates = ([prefer] if prefer else []) + [
        f"npu:{i}" for i in range(torch.npu.device_count())
    ]
    for d in candidates:
        try:
            torch.npu.set_device(d)
            x = torch.randn(64, 64, device=d).half()
            _ = x @ x
            torch.npu.synchronize()
            return d
        except Exception:
            continue
    return None


def on_npu() -> bool:
    """True iff a healthy, idle NPU is available (cheap; reuses :func:`pick_device`)."""
    return pick_device() is not None
