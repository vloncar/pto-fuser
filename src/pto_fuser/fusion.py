"""Fusion decision procedure — the staged-vs-fused decision (design §4, §6, §8).

Graph capture is the default backend: a staged chain replayed as one dispatch,
perf-neutral-to-winning everywhere. Lever 6 (a single hand-fused kernel) is the
*last resort* — design §4 keeps it "only where launch-bound small-`T` justifies it
and graph capture is insufficient." What graph capture does **not** remove is the
HBM round-trip of intermediates between stages: the resident state ``S`` of the
chunk scan, or the qk matrix of kkt. A fused kernel keeps those on-chip. Whether
that beats staged-captured is an *empirical, per-stage* question — so the fusion decision's
deliverable is a measured decision, not a blanket "fuse everything."

`decide` runs both lowerings of one stage on identical inputs and returns a
`FusionDecision`:

  * **frob gate** — the fused output must match the staged output (design §6: a
    broken/garbage fused pipeline fails here; ungated "wins" were the historical trap).
  * **determinism gate** — the fused kernel run twice must be bit-identical (design
    §6: mandatory on any fused/mega/scan lowering — this is what caught the mega
    H=64 non-determinism a single-run frob check masks).
  * **measurement** — both backends timed back-to-back + one trailing sync (what
    exposes the dispatch + the HBM traffic).

The fused lowering is **kept only if** gated-green, deterministic, *and* faster.
Otherwise the staged-captured lowering stands — exactly the lever ordering rule
(prefer staged+captured; reach for a fused kernel only on a measured win).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from .gate import frob_rel, gate_determinism

Outputs = Dict[str, torch.Tensor]
RunFn = Callable[[], Outputs]


@dataclass
class FusionDecision:
    """The measured verdict for one fused stage vs its staged-captured lowering."""
    stage: str
    kernel: str                 # the fused-kernel registry key
    gated_ok: bool              # max frob_rel(fused, staged) over outputs < tol
    deterministic: bool         # fused run twice, bit-identical
    faster: bool                # fused ms/call < staged ms/call
    kept: bool                  # gated_ok and deterministic and faster
    t_staged_ms: float
    t_fused_ms: float
    frob: float                 # worst per-output frob_rel
    detail: str

    @property
    def speedup(self) -> float:
        return self.t_staged_ms / self.t_fused_ms if self.t_fused_ms else float("nan")

    def __str__(self) -> str:
        verdict = "FUSE" if self.kept else "stage"
        flags = f"frob={self.frob:.1e} {'det' if self.deterministic else 'NDET'}"
        return (f"[{verdict}] {self.stage:<14} {self.kernel:<14} {flags:<18} "
                f"staged {self.t_staged_ms:7.3f}ms -> fused {self.t_fused_ms:7.3f}ms "
                f"({self.speedup:.2f}x)")


def _time(fn: RunFn, iters: int = 30, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3       # ms/call


def decide(stage: str, kernel: str, staged: RunFn, fused: RunFn, *,
           tol: float = 2e-2, iters: int = 30, warmup: int = 5) -> FusionDecision:
    """Gate + measure the fused lowering of one stage against staged-captured.

    `staged` and `fused` are zero-arg callables returning name->tensor dicts over
    the *same* inputs (the staged-captured backend and the FusedNode backend). The
    fused outputs are gated against the staged outputs (the correctness reference),
    the fused kernel is checked deterministic, and both are timed.
    """
    ref = staged()
    got = fused()
    worst = max((frob_rel(got[n], ref[n]) for n in ref if n in got), default=float("inf"))
    gated_ok = worst < tol

    det = gate_determinism(fused)

    t_staged = _time(staged, iters=iters, warmup=warmup)
    t_fused = _time(fused, iters=iters, warmup=warmup)
    faster = t_fused < t_staged
    kept = gated_ok and det.passed and faster
    detail = (f"frob_rel={worst:.2e} (tol {tol:.0e}); {det.detail}; "
              f"{'faster' if faster else 'not faster'}")
    return FusionDecision(stage, kernel, gated_ok, det.passed, faster, kept,
                          t_staged, t_fused, worst, detail)


def format_decisions(decisions: List[FusionDecision]) -> str:
    kept = sum(d.kept for d in decisions)
    lines = [str(d) for d in decisions]
    lines.append(f"-- {kept}/{len(decisions)} stages fused "
                 f"(gated-green, deterministic, and faster than staged-captured)")
    return "\n".join(lines)
