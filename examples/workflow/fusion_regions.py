"""Workflow demo: fusion-region identification (toward megakernel generation).

The first megakernel step (docs/DESIGN.md §8) as an **analysis only** — no codegen, no
device. It partitions the canonical GDN / KDA forward into the maximal scopes a single
kernel could subsume, cut at the opaque triangular-inverse boundary, and scores each by
the HBM traffic that fusing it would keep on-chip (the L2-residency opportunity). The
sizing is a pure meta-tensor shape walk, so this runs off-NPU.

    python examples/workflow/fusion_regions.py --H 16 --nc 8
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "examples"))
sys.path.insert(0, os.path.join(_ROOT, "src"))                # pto_fuser (no install needed)

from pto_fuser import identify_fusion_regions  # noqa: E402
from attention._gdn_full import (build_gdn_program, make_gdn_inputs,  # noqa: E402
                                 prepare_gdn_bindings)
from attention._kda_full import (build_kda_program, make_kda_inputs,  # noqa: E402
                                 prepare_kda_bindings)


def show(name, build, mk, prep, B, H, nc, C, D):
    inp = mk(B, H, nc, C, D, "cpu")
    binds = prep(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g_in"])
    canon = build(B, H, nc, C, D, D ** -0.5)
    an = identify_fusion_regions(canon, binds)
    print(f"\n=== {name}  B{B} H{H} nc{nc} C{C} D{D} ===")
    print(an)
    for r in an.ranked():
        print(f"\n  region {r.id} [{r.kind}] — {r.n_einsum} contractions, "
              f"{r.n_glue} glue, {len(r.internal)} internal tensors kept on-chip")
        print(f"    reads : {', '.join(r.inputs)}")
        print(f"    writes: {', '.join(r.outputs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--nc", type=int, default=8)
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--D", type=int, default=128)
    args = ap.parse_args()
    B = 1
    show("GDN", build_gdn_program, make_gdn_inputs, prepare_gdn_bindings,
         B, args.H, args.nc, args.C, args.D)
    show("KDA", build_kda_program, make_kda_inputs, prepare_kda_bindings,
         B, args.H, args.nc, args.C, args.D)


if __name__ == "__main__":
    main()
