"""M2 planner — gate-and-measure lever selection over the public substrate knobs.

Design §4 levers 2 and 3 (read-mode NT/NN-strided/TN, and operand-swap → fused
store) are realized **inside the soft-frozen substrate**: it auto-selects them from
the equation+layout and exposes the documented toggles ``EINSUM_DISABLE_NT`` /
``EINSUM_DISABLE_OPERAND_SWAP``. Per design §2 the fuser *selects among* these
substrate capabilities; it does not re-implement them. So the planner, for each
distinct ``EinsumNode`` contraction:

  1. measures the substrate's optimized lowering against the always-valid Phase-A
     NN baseline on that node's real operands;
  2. **frob-gates** the two equivalent (design §6 — a broken lowering that produced
     zero/garbage would fail here);
  3. keeps the optimization only when gated-green **and** faster than the baseline
     (design §8 M2: "kept only where gated-green and faster than default").

The output is an annotated ``Program`` (each ``EinsumNode``'s ``read_mode`` /
``fuse_out`` pinned to the kept choice) plus a per-node ``LeverDecision`` ledger.
Distinct contractions are measured once and reused (the DeltaNet scan repeats the
same two shapes ``nc`` times), so planning a 600-node program costs a handful of
builds.

Lever 4 (glue absorption) is a *detector* here (`absorption_candidates`): it finds
``VecGlueNode`` → ``EinsumNode`` adjacencies whose intermediate round-trips HBM and
could be folded into the contraction's epilogue/prologue. The actual on-chip fold
is the fused-node backend (design §5) — the one place the fuser emits new device
code — and is staged for M4 with the ``kkt_fused`` prototype as its evidence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .executor import StagedExecutor, substrate_modes, _load_substrate_einsum
from .gate import frob_rel
from .ir import EinsumNode, Program, VecGlueNode


@dataclass
class LeverDecision:
    """The measured verdict for one lever on one distinct contraction."""
    node: str           # the EinsumNode output name (representative of the shape class)
    equation: str
    lever: str          # "direct_read" | "operand_swap"
    fired: bool         # the substrate actually changed the lowering vs forced-NN
    gated_ok: bool      # frob_rel(candidate, baseline) < tol
    faster: bool        # candidate wall-clock < baseline
    kept: bool          # fired and gated_ok and faster
    t_base_us: float
    t_cand_us: float
    detail: str

    def __str__(self) -> str:
        verdict = "KEEP" if self.kept else "drop"
        speed = (self.t_base_us / self.t_cand_us) if self.t_cand_us else float("nan")
        return (f"[{verdict}] {self.equation:<22} {self.lever:<13} "
                f"{self.detail:<22} {self.t_base_us:7.1f}->{self.t_cand_us:7.1f}us "
                f"({speed:.2f}x)")


# --------------------------------------------------------------------------- #
#  measurement
# --------------------------------------------------------------------------- #
@dataclass
class _Measurement:
    t_us: float
    out: torch.Tensor
    in_nt: int
    out_fusible: int
    swapped: bool


class Planner:
    """Selects + gates the read-mode / fused-store levers per EinsumNode.

    `measure` is injectable so the keep-logic is testable off-NPU; the default
    builds + times the real substrate kernel.
    """

    def __init__(self, executor: Optional[StagedExecutor] = None, *, tol: float = 2e-2,
                 warmup: int = 3, iters: int = 20,
                 measure: Optional[Callable] = None) -> None:
        self.executor = executor or StagedExecutor()
        self.tol = tol
        self.warmup = warmup
        self.iters = iters
        self._measure = measure or self._measure_substrate
        self._Builder = None

    # -- public API --------------------------------------------------------- #
    def plan(self, program: Program, bindings: Dict[str, torch.Tensor]
             ) -> Tuple[Program, List[LeverDecision]]:
        """Return (annotated program, decision ledger).

        Runs the program once (staged, auto modes) to materialize every node's real
        operands, then measures each *distinct* contraction and pins the kept levers.
        """
        env = self.executor.run(program, bindings, return_env=True)
        cache: Dict[tuple, Dict[str, LeverDecision]] = {}
        decisions: List[LeverDecision] = []
        chosen: Dict[int, Dict[str, bool]] = {}

        for idx, node in enumerate(program.nodes):
            if not isinstance(node, EinsumNode):
                continue
            a, b = (env[n].contiguous() for n in node.inputs)
            key = (node.equation, tuple(a.shape), tuple(b.shape), a.dtype)
            if key not in cache:
                cache[key] = self._evaluate(node, a, b)
                decisions.extend(cache[key].values())
            verds = cache[key]
            chosen[idx] = {"direct_read": verds["direct_read"].kept,
                           "operand_swap": verds["operand_swap"].kept}

        annotated = self._annotate(program, chosen)
        return annotated, decisions

    def absorption_candidates(self, program: Program) -> List[Tuple[str, str]]:
        """Lever-4 detector: (glue_output, einsum_output) pairs where a VecGlueNode
        feeds straight into an EinsumNode (an HBM round-trip the fused-node backend
        could fold into an epilogue/prologue). Detection only — codegen is M4."""
        producer = {}
        for node in program.nodes:
            producer[node.output] = node
        pairs = []
        for node in program.nodes:
            if isinstance(node, EinsumNode):
                for src in node.inputs:
                    p = producer.get(src)
                    if isinstance(p, VecGlueNode):
                        pairs.append((p.output, node.output))
        return pairs

    # -- internals ---------------------------------------------------------- #
    def _evaluate(self, node: EinsumNode, a, b) -> Dict[str, LeverDecision]:
        eq = node.equation
        base = self._measure(eq, a, b, read_mode="NN", fuse_out=False)
        dr = self._measure(eq, a, b, read_mode="auto", fuse_out=False)   # NT/NN/TN only
        sw = self._measure(eq, a, b, read_mode="NN", fuse_out=True)      # swap only

        out = {}
        out["direct_read"] = self._verdict(node, eq, "direct_read", base, dr,
                                           fired=dr.in_nt != 0,
                                           label=_mode_name(dr.in_nt))
        out["operand_swap"] = self._verdict(node, eq, "operand_swap", base, sw,
                                            fired=sw.swapped,
                                            label="swap+fused" if sw.swapped else "no-swap")
        return out

    def _verdict(self, node, eq, lever, base: _Measurement, cand: _Measurement,
                 *, fired: bool, label: str) -> LeverDecision:
        rel = frob_rel(cand.out, base.out)
        gated = rel < self.tol
        faster = cand.t_us < base.t_us
        kept = fired and gated and faster
        detail = f"{label} frob={rel:.1e}"
        return LeverDecision(node.output, eq, lever, fired, gated, faster, kept,
                             base.t_us, cand.t_us, detail)

    def _annotate(self, program: Program, chosen: Dict[int, Dict[str, bool]]) -> Program:
        nodes = []
        for idx, node in enumerate(program.nodes):
            if isinstance(node, EinsumNode) and idx in chosen:
                c = chosen[idx]
                node = replace(node,
                               read_mode="auto" if c["direct_read"] else "NN",
                               fuse_out=bool(c["operand_swap"]))
            nodes.append(node)
        return Program(nodes=nodes, inputs=list(program.inputs),
                       outputs=list(program.outputs))

    def _builder(self):
        if self._Builder is None:
            _load_substrate_einsum()              # ensures pto-einsum is importable
            from pto_einsum import EinsumBuilder   # noqa: import-after-path
            self._Builder = EinsumBuilder
        return self._Builder

    def _measure_substrate(self, eq, a, b, *, read_mode, fuse_out) -> _Measurement:
        Builder = self._builder()
        with substrate_modes(read_mode, fuse_out):
            builder = Builder(eq, [tuple(a.shape), tuple(b.shape)], a.dtype)
            runner = builder.build()
            for _ in range(self.warmup):
                out = runner(a, b)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(self.iters):
                out = runner(a, b)
            torch.npu.synchronize()
            t_us = (time.perf_counter() - t0) / self.iters * 1e6
        r = builder.recipe
        return _Measurement(t_us, out, int(r["in_nt"]), int(r["out_fusible"]),
                            bool(builder._swap_operands))


def _mode_name(in_nt: int) -> str:
    return {0: "phaseA-NN", 1: "NT", 2: "NN-strided", 3: "TN"}.get(in_nt, f"in_nt={in_nt}")


def format_decisions(decisions: List[LeverDecision]) -> str:
    kept = sum(d.kept for d in decisions)
    lines = [str(d) for d in decisions]
    lines.append(f"-- {kept}/{len(decisions)} levers kept (gated-green and faster)")
    return "\n".join(lines)
