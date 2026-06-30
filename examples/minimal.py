"""The smallest useful pto-fuser program.

Builds a two-stage IR Program — a batched matmul contraction followed by a scaled,
causal-masked epilogue — and runs it on the staged executor. This is the "hello
world": it shows the three things every pto-fuser program needs, with no knobs.

    1. describe the computation as a list of typed nodes (an EinsumNode for the
       contraction, VecGlueNodes for the elementwise glue),
    2. wrap them in a Program with named inputs/outputs,
    3. run it with StagedExecutor over a {name: tensor} binding.

    python examples/minimal.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # examples/
import common  # noqa: E402  (puts src/ on the path, sets PTO_LIB_PATH)

import torch  # noqa: E402

from pto_fuser import (EinsumNode, Program, StagedExecutor, VecGlueNode,  # noqa: E402
                       frob_rel)


def build_program(scale: float) -> Program:
    """attn = scale · tril( q @ kᵀ ), batched over n.  q,k: [n, t, d] → out: [n, t, t]."""
    return Program(
        nodes=[
            EinsumNode("ntd,nsd->nts", ["q", "k"], "qk", out_dtype=torch.float16),
            VecGlueNode("tril", ["qk"], "qk_masked", params={"diagonal": 0}),
            VecGlueNode("scale", ["qk_masked"], "attn", params={"scalar": scale},
                        out_dtype=torch.float16),
        ],
        inputs=["q", "k"],
        outputs=["attn"],
    )


def main():
    dev = common.pick_device()
    if dev is None:
        prog = build_program(0.125)
        print("no healthy NPU — Program builds off-NPU:")
        print(f"  {len(prog.nodes)} nodes, inputs={prog.inputs}, outputs={prog.outputs}")
        return

    n, t, d = 2, 64, 128
    scale = d ** -0.5
    q = torch.randn(n, t, d, device=dev, dtype=torch.float16)
    k = torch.randn(n, t, d, device=dev, dtype=torch.float16)

    out = StagedExecutor().run(build_program(scale), {"q": q, "k": k})["attn"]

    ref = (torch.tril(torch.einsum("ntd,nsd->nts", q.float(), k.float())) * scale)
    print(f"attn shape: {tuple(out.shape)}   frob_rel vs torch: {frob_rel(out, ref):.2e}")


if __name__ == "__main__":
    main()
