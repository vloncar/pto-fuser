"""Workflow demo: read-mode / fused-output selection (the Planner).

The Planner runs each contraction stage in its always-valid baseline lowering and in
the library's direct-read / operand-swap lowerings, gates each candidate against the
baseline (frob ≡), times them, and keeps a candidate only when it both gates green
and runs faster. It also reports the glue→einsum adjacencies that the fused-node
backend can absorb.

    python examples/workflow/read_modes.py --B 8 --H 32 --nc 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # examples/
import common  # noqa: E402

import torch  # noqa: E402

from pto_fuser import Planner, StagedExecutor, format_decisions, gate_outputs  # noqa: E402
from pto_fuser.forwards import (build_deltanet_program, deltanet_reference,  # noqa: E402
                                gdn_contraction_stages, make_inputs)


def deltanet(dev, B, H, nc, C, D):
    print(f"\n=== DeltaNet  B{B} H{H} nc{nc} C{C} D{D} ===")
    scale = D ** -0.5
    program = build_deltanet_program(B, H, nc, C, D, scale)
    bindings = make_inputs(B, H, nc, C, D, dev)
    planner = Planner()
    annotated, decisions = planner.plan(program, bindings)
    print(format_decisions(decisions))
    ref = deltanet_reference(**bindings, B=B, H=H, nc=nc, C=C, D=D, scale=scale)
    got = StagedExecutor().run(annotated, bindings)
    bad = [str(r) for r in gate_outputs(got, ref, tol=2e-2) if not r.passed]
    print("annotated-program gate:", "ALL OK" if not bad else "FAIL\n" + "\n".join(bad))
    print("glue-absorption candidates:", planner.absorption_candidates(program))


def gdn(dev, B, H, nc, C, D):
    print(f"\n=== GDN contraction stages  B{B} nc{nc} H{H} C{C} D{D} ===")
    planner = Planner()
    for name, prog, bindings in gdn_contraction_stages(B=B, nc=nc, H=H, C=C, D=D, device=dev):
        _, decisions = planner.plan(prog, bindings)
        for d in decisions:
            print(f"  {name:<9} {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--nc", type=int, default=8)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--D", type=int, default=128)
    args = ap.parse_args()
    dev = common.pick_device()
    if dev is None:
        print("no healthy NPU — this demo needs an Ascend device.")
        return
    torch.manual_seed(0)
    deltanet(dev, args.B, args.H, args.nc, args.C, args.D)
    gdn(dev, args.B, args.H, args.nc, args.C, args.D)


if __name__ == "__main__":
    main()
