"""Wall-clock timing and table formatting.

The timing convention matches the one the fusion decision uses internally: launch
the candidate ``iters`` times back-to-back and synchronize **once** at the end, so
the measurement captures host dispatch overhead (the thing graph capture removes)
rather than hiding it behind a per-call sync. A few warmup iterations prime the
kernel build / workspace setup / caches first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence


def time_ms(fn: Callable[[], object], iters: int = 30, warmup: int = 5) -> float:
    """Median-free mean ms/call: ``warmup`` untimed calls, then ``iters`` timed
    back-to-back with a single trailing synchronize."""
    import torch

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


@dataclass
class Measurement:
    """One named timing, optionally relative to a baseline.

    ``speedup`` is filled in by :meth:`relative_to`; ``note`` carries a gate verdict
    or any short annotation the table should show.
    """
    name: str
    ms: float
    speedup: Optional[float] = None
    note: str = ""

    def relative_to(self, baseline_ms: float) -> "Measurement":
        self.speedup = baseline_ms / self.ms if self.ms > 0 else float("nan")
        return self


def format_table(rows: Sequence[Measurement], *, title: str = "",
                 baseline: Optional[str] = None) -> str:
    """Render measurements as a GitHub-flavoured markdown table.

    If ``baseline`` names a row, every row's ``speedup`` is computed against it (so
    callers can pass raw timings and let the table do the division).
    """
    rows = list(rows)
    if baseline is not None:
        base = next((r.ms for r in rows if r.name == baseline), None)
        if base is not None:
            for r in rows:
                r.relative_to(base)

    lines: List[str] = []
    if title:
        lines.append(f"### {title}")
        lines.append("")
    lines.append("| variant | ms/call | speedup | note |")
    lines.append("|---------|--------:|--------:|------|")
    for r in rows:
        sp = "—" if r.speedup is None else f"{r.speedup:.2f}×"
        lines.append(f"| {r.name} | {r.ms:.3f} | {sp} | {r.note} |")
    return "\n".join(lines)
