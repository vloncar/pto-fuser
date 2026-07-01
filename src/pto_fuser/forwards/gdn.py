"""GDN contraction stages — the second reference forward for the planner.

A second reference forward (GDN or KDA) confirms the planner generalizes. GDN shares the kkt/wy/chunk_h/chunk_o stages with
DeltaNet but in a **different equation family** — the head axis ``h`` is a
non-innermost batch axis, which is exactly what drives the library's NT /
NN-strided / TN direct reads (the read/store modes–2.13). The full GDN forward additionally needs
the gating cumsum, GQA repeat, and the chunk_h cross-chunk recurrence with resident
state (the resident-state feature); those are glue around the same four contractions,
so for the planner — whose job is the read-mode / fused-store decision per contraction
— the four contraction stages are the unit that exercises every read mode.

The four stages (regular-nc, chunk loop folded into the leading batch ``b = B*nc``;
shapes mirror ``benchmarks/complex/gdn/einsum_gdn.py``):

  * ``kkt``     "bihd,bjhd->bihj"   contraction d innermost on both -> NT
  * ``wy_fast`` "bihj,bjhd->bihd"   free1 d innermost on B, head-strided K -> NN-strided
  * ``chunk_h`` "bvhd,bvhe->bhde"   contraction v outer on both -> TN
  * ``chunk_o`` "bvhd,bhde->bvhe"   head-strided -> NN-strided

Each is returned as a single-``EinsumNode`` ``Program`` with synthetic contiguous
fp16 operands, so ``Planner.plan`` runs over them exactly as over the DeltaNet stages.
"""
from __future__ import annotations

import torch

from ..ir import EinsumNode, Program


def _stage(name: str, eq: str) -> Program:
    a, b = (s.strip() for s in eq.split("->")[0].split(","))
    out = eq.split("->")[1].strip()
    return Program(nodes=[EinsumNode(eq, [a, b], out, out_dtype=torch.float16)],
                   inputs=[a, b], outputs=[out])


def gdn_contraction_stages(B: int = 2, nc: int = 4, H: int = 16, C: int = 64,
                           D: int = 128, device="npu:0") -> list:
    """Return [(stage_name, Program, bindings), ...] for the four GDN contractions.

    Operand names match the equation indices so each single-node Program binds its
    two inputs by name. Shapes use the regular-nc fold (leading batch Bn = B*nc).
    """
    Bn = B * nc
    g = dict(device=device, dtype=torch.float16)
    rand = lambda *shape: torch.randn(*shape, **g)

    stages = []

    # kkt: "bihd,bjhd->bihj"  — both operands [Bn, C, H, D]; contraction d innermost.
    prog = _stage("kkt", "bihd,bjhd->bihj")
    stages.append(("kkt", prog,
                   {"bihd": rand(Bn, C, H, D), "bjhd": rand(Bn, C, H, D)}))

    # wy_fast: "bihj,bjhd->bihd"  — A_inv [Bn,C,H,C], stacked kv [Bn,C,H,D].
    prog = _stage("wy_fast", "bihj,bjhd->bihd")
    stages.append(("wy_fast", prog,
                   {"bihj": rand(Bn, C, H, C), "bjhd": rand(Bn, C, H, D)}))

    # chunk_h kv: "bvhd,bvhe->bhde"  — k [Bn,C,H,D], v_scaled [Bn,C,H,D]; contract v.
    prog = _stage("chunk_h", "bvhd,bvhe->bhde")
    stages.append(("chunk_h", prog,
                   {"bvhd": rand(Bn, C, H, D), "bvhe": rand(Bn, C, H, D)}))

    # chunk_o qh: "bvhd,bhde->bvhe"  — q [Bn,C,H,D], state h [Bn,H,D,D].
    prog = _stage("chunk_o", "bvhd,bhde->bvhe")
    stages.append(("chunk_o", prog,
                   {"bvhd": rand(Bn, C, H, D), "bhde": rand(Bn, H, D, D)}))

    return stages
