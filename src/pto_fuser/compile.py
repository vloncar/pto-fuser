"""Compile driver — propose / verify / dispose.

`compile_program` is the top of the separated stack: it canonicalizes the program
to the always-valid NN baseline, asks the `Policy` for a transform pipeline, and then
applies each transform under the correctness+performance gate:

  * **propose** — the policy (over the cost model) picks *which* transforms to try and
    *in what order*;
  * **verify** — each proposed rewrite is run on device, its outputs frob-gated
    against the **canonical** reference (never against the previous candidate — the
    floor is always the correct staged lowering), checked deterministic, and timed
    against the current best;
  * **dispose** — keep the transform iff it is gated-green, deterministic, *and*
    faster than the program without it; otherwise roll back.

The result is the lowered `Program` plus a `CompilationReport` recording every
verdict. This is the same discipline the old per-lever `decide()` calls enforced,
but hoisted into one loop over pure transforms — so the heuristic (policy/cost),
the transformation (the passes), and the verification (this gate) are three separable
pieces instead of one gated build-flag branch.

Off device (or with ``verify=False`` / no ``bindings``) it still runs the policy and
applies the transforms, producing the lowered program and an *unverified* report —
which is exactly what the structural unit tests exercise.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

from .cost import Features
from .gate import frob_rel, gate_determinism
from .ir import Program
from .policy import Policy
from .report import CompilationReport, TransformRecord
from .transform import canonicalize


@dataclass
class CompileResult:
    program: Program
    report: CompilationReport


def compile_program(program: Program, features: Features, *,
                    policy: Optional[Policy] = None,
                    bindings: Optional[Dict] = None,
                    verify: bool = True,
                    tol: float = 2e-2, iters: int = 30, warmup: int = 5
                    ) -> CompileResult:
    """Lower ``program`` under the propose/verify/dispose loop. See module docstring."""
    policy = policy or Policy()
    canonical = canonicalize(program)
    plan = policy.pipeline(canonical, features)

    do_verify = verify and bindings is not None
    records = []
    prog = canonical

    if do_verify:
        ref_out, base_ms, _ = _run_and_time(canonical, bindings, iters, warmup)
    for planned in plan:
        t = planned.transform
        pred = planned.prediction
        rec = TransformRecord(name=t.name, summary=t.summary,
                              predicted_benefit=pred.benefit,
                              predicted_rationale=pred.rationale,
                              matched=t.match(prog) > 0, attempted=False, kept=False)
        if not rec.matched:
            rec.note = "no match after prior transforms"
            records.append(rec)
            continue
        cand = t.apply(prog).program

        if not do_verify:
            prog = cand
            rec.attempted = True
            rec.kept = True
            rec.note = "applied (unverified)"
            records.append(rec)
            continue

        got, cand_ms, replay = _run_and_time(cand, bindings, iters, warmup)
        worst = max((frob_rel(got[n], ref_out[n]) for n in ref_out if n in got),
                    default=float("inf"))
        det = gate_determinism(replay)
        faster = cand_ms < base_ms
        kept = worst < tol and det.passed and faster
        rec.attempted = True
        rec.kept = kept
        rec.frob = worst
        rec.deterministic = det.passed
        rec.faster = faster
        rec.t_base_ms = base_ms
        rec.t_cand_ms = cand_ms
        if kept:
            prog = cand
            base_ms = cand_ms       # subsequent transforms must beat the new best
        records.append(rec)

    return CompileResult(prog, CompilationReport(features, records, do_verify))


# --------------------------------------------------------------------------- #
#  device helpers (imported lazily so the module loads off-NPU)
# --------------------------------------------------------------------------- #
def _run_and_time(program: Program, bindings: Dict, iters: int, warmup: int):
    """Capture ``program`` as one graph; return (outputs, ms/call, replay_fn) where
    ``replay_fn`` runs one cloned replay of the *same* captured graph (so the
    determinism gate re-runs this candidate, not a fresh capture)."""
    import torch
    from .graph import GraphReplayExecutor
    gr = GraphReplayExecutor().capture(program, bindings)
    for _ in range(warmup):
        out = gr.replay(bindings)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        gr.replay(bindings, clone=False)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    return out, ms, (lambda: gr.replay(bindings))
