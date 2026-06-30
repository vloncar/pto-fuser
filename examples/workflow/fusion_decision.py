"""Workflow demo: the staged-vs-fused fusion decision.

For each candidate stage the decision procedure gates the fused lowering against the
staged one (frob ≡ + determinism) and times both, keeping the fused kernel only when
it gates green, is deterministic, AND is faster — otherwise the staged-captured
lowering stands. Two stages, the two fusion features:

  * ``chunk_h_scan`` — resident state (the carried state stays on-chip across chunks);
  * ``kkt_gated``    — glue absorption (the gated/masked epilogue folded into the store).

    python examples/workflow/fusion_decision.py --B 1 --H 4 --nc 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # examples/
import common  # noqa: E402

import torch  # noqa: E402

from pto_fuser import GraphReplayExecutor, decide  # noqa: E402
from pto_fuser.forwards import (build_kkt_fused_program, build_scan_fused_program,  # noqa: E402
                                build_scan_staged_program, kkt_reference,
                                make_kkt_inputs, make_scan_inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--nc", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    dev = common.pick_device()
    if dev is None:
        print("no healthy NPU — this demo needs an Ascend device.")
        return
    torch.manual_seed(0)
    B, H, nc = args.B, args.H, args.nc

    scan_in = make_scan_inputs(B, H, nc, dev)
    gs = GraphReplayExecutor().capture(build_scan_staged_program(B, H, nc), scan_in)
    gf = GraphReplayExecutor().capture(build_scan_fused_program(B, H, nc), scan_in)
    d_scan = decide("chunk_h_scan", "chunk_h_scan",
                    lambda: gs.replay(scan_in, clone=False),
                    lambda: gf.replay(scan_in, clone=False), tol=2e-2, iters=args.iters)

    kkt_in = make_kkt_inputs(nc, H, dev)
    kref = kkt_reference(kkt_in, nc, H)
    gk = GraphReplayExecutor().capture(build_kkt_fused_program(nc, H), kkt_in)
    d_kkt = decide("kkt_gated", "kkt_gated",
                   lambda: {n: kref[n].clone() for n in kref},
                   lambda: gk.replay(kkt_in, clone=False), tol=2e-2, iters=args.iters)

    print(f"\nfusion decisions  B={B} H={H} nc={nc}")
    for d in (d_scan, d_kkt):
        print("  " + str(d))


if __name__ == "__main__":
    main()
