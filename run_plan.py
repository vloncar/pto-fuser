"""M2 driver — run the read-mode / fused-store Planner over the reference forwards.

Measures levers 2/3 on each distinct DeltaNet and GDN contraction stage, prints the
gate-and-measure decision ledger (which lowering fired, gated frob vs the Phase-A
baseline, the wall-clock, and the keep/drop verdict), gate-checks that the annotated
program still matches the fp32 reference, and lists the lever-4 absorption candidates.

    export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
    python run_plan.py --B 8 --H 32 --nc 8 --C 64 --D 128
"""
import argparse
import os
import sys

os.environ.setdefault("PTO_LIB_PATH", "/home/vloncar/work/einsum_workspace/pto-isa")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch

try:
    import torch_npu  # noqa
except ImportError:
    print("torch_npu not available — this driver needs an Ascend NPU.")
    sys.exit(1)

from pto_fuser import (Planner, StagedExecutor, format_decisions, gate_outputs)
from pto_fuser.forwards import (build_deltanet_program, deltanet_reference,
                                gdn_contraction_stages, make_inputs)


def plan_deltanet(args):
    print(f"\n=== DeltaNet  B{args.B} H{args.H} nc{args.nc} C{args.C} D{args.D} ===")
    dev = torch.device("npu:0")
    scale = args.D ** -0.5
    program = build_deltanet_program(args.B, args.H, args.nc, args.C, args.D, scale)
    bindings = make_inputs(args.B, args.H, args.nc, args.C, args.D, dev)

    planner = Planner()
    annotated, decisions = planner.plan(program, bindings)
    print(format_decisions(decisions))

    # the annotated (lever-pinned) program must still match the fp32 reference.
    ref = deltanet_reference(**bindings, B=args.B, H=args.H, nc=args.nc,
                             C=args.C, D=args.D, scale=scale)
    got = StagedExecutor().run(annotated, bindings)
    results = gate_outputs(got, ref, tol=2e-2)
    bad = [str(r) for r in results if not r.passed]
    print("annotated-program gate:", "ALL OK" if not bad else "FAIL\n" + "\n".join(bad))

    pairs = planner.absorption_candidates(program)
    print(f"lever-4 absorption candidates (glue->einsum, M4 fused-node): {pairs}")
    return not bad


def plan_gdn(args):
    print(f"\n=== GDN contraction stages  B{args.B} nc{args.nc} H{args.H} "
          f"C{args.C} D{args.D} ===")
    planner = Planner()
    ok = True
    for name, prog, bindings in gdn_contraction_stages(
            B=args.B, nc=args.nc, H=args.H, C=args.C, D=args.D):
        _, decisions = planner.plan(prog, bindings)
        for d in decisions:
            print(f"  {name:<9} {d}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--nc", type=int, default=8)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--D", type=int, default=128)
    args = ap.parse_args()
    torch.npu.set_device("npu:0")
    torch.manual_seed(0)

    ok = plan_deltanet(args)
    plan_gdn(args)
    print("\n" + ("ALL OK" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
