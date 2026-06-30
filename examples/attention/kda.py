"""KDA (Kimi Delta Attention) as a pto-fuser forward.

KDA is GDN with a **per-dimension** gate instead of a per-head scalar one. The key
implementation insight (and why it needs *no new framework capability*) is that the
per-dim gate is baked into the operands — the decayed keys/queries are formed on the
host, after which KDA runs the **identical** einsum equations as GDN: same kkt, same
WY recompute, same cross-chunk scan, same output. So the same Programs and the same
two fused stages apply; only the operand preparation differs.

This example reuses the GDN demo to make that concrete: the staged backbone and the
two fused-stage decisions are exactly GDN's — the per-dim gate would be applied as a
Vec scaling on k/q before the contraction (a proven glue class), leaving the graph
structure unchanged.

    python examples/attention/kda.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from attention.gdn import backbone, fused_stage_decisions  # noqa: E402
from common import pick_device  # noqa: E402
from pto_fuser.forwards import build_deltanet_program  # noqa: E402


def main():
    dev = pick_device()
    print("KDA = GDN + per-dimension gate (folded into operands); same einsum graph.")
    if dev is None:
        print("no healthy NPU — building the (shared) Program off-NPU to check it constructs.")
        build_deltanet_program(2, 4, 4, 64, 128, 128 ** -0.5)
        print("Program built OK.")
        return
    torch.manual_seed(0)
    # Same graph as GDN; the per-dim gate is an operand-prep Vec scaling (not shown
    # here) that does not change the contraction equations or the fused stages.
    backbone(dev)
    fused_stage_decisions(dev)


if __name__ == "__main__":
    main()
