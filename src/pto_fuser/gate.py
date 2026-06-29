"""Correctness gating — non-negotiable (docs/FUSER_DESIGN.md §6).

Every non-default lowering is kept only if it passes, against the default staged
lowering on identical inputs:

  * **frob_rel gate** — relative Frobenius norm of the difference under a fixed
    tolerance. Staged-vs-fused is *always* gated; ungated "wins" turned out to be
    broken (zero/garbage) pipelines once a frob check was added.
  * **determinism gate** — run the candidate twice on identical input; bit-identical
    required. This caught the mega H=64 non-determinism (a missing cumsum Vec->Scalar
    edge) that a single-run frob check masks. Mandatory on any mega/fused/scan
    lowering.

The gate is part of the planner loop, not a separate test phase: a lever that does
not pass is discarded and the default lowering stands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch


def frob_rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    g, r = got.float(), ref.float()
    return (g - r).norm().item() / r.norm().clamp_min(1e-9).item()


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'OK' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def gate_frob_rel(name: str, got: torch.Tensor, ref: torch.Tensor,
                  tol: float) -> GateResult:
    rel = frob_rel(got, ref)
    amax = (got.float() - ref.float()).abs().max().item()
    return GateResult(name, rel < tol,
                      f"frob_rel={rel:.3e} (tol {tol:.1e})  max_abs={amax:.3e}")


def gate_outputs(got: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor],
                 tol: float) -> list[GateResult]:
    """frob_rel-gate every named tensor present in both dicts."""
    results = []
    for name in ref:
        if name in got:
            results.append(gate_frob_rel(name, got[name], ref[name], tol))
    return results


def gate_determinism(run_fn: Callable[[], Dict[str, torch.Tensor]]) -> GateResult:
    """Run twice on identical input; require every output bit-identical."""
    a = run_fn()
    b = run_fn()
    mismatches = []
    for name in a:
        if not torch.equal(a[name], b[name]):
            diff = (a[name].float() - b[name].float()).abs().max().item()
            mismatches.append(f"{name}(Δ={diff:.3e})")
    passed = not mismatches
    detail = "bit-identical across 2 runs" if passed else "NDET: " + ", ".join(mismatches)
    return GateResult("determinism", passed, detail)
