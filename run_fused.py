"""M4 driver — the fusion decision: staged-captured vs a single fused kernel.

For each candidate stage (the resident-state chunk scan = lever 5, the gated kkt =
lever 6) this builds both lowerings over identical inputs, runs them through the
graph-replay backend, and prints the gated, measured decision (frob ≡ staged +
determinism + speed). The fused lowering is *kept only* where it gates green, is
deterministic, and is faster than staged-captured — design §4 lever 6 ("last resort,
narrow regime only"), §8 M4 ("a documented decision per stage, with the measurement
that chose it").

  export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
  export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
  python run_fused.py --B 8 --H 32 --nc 8         # production-ish; both stages
  python run_fused.py --B 1 --H 2 --nc 4 --stage scan
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from pto_fuser import (GraphReplayExecutor, decide, format_fusion_decisions,  # noqa: E402
                       frob_rel, gate_outputs)
from pto_fuser.forwards import (build_kkt_fused_program, build_scan_fused_program,  # noqa: E402
                                build_scan_staged_program, kkt_reference,
                                make_kkt_inputs, make_scan_inputs, scan_reference)


def healthy_device() -> str:
    """Pick the first NPU that passes a tiny matmul (the box is shared; a chip
    pinned by a neighbor job, or left wedged by an aicore timeout, is skipped)."""
    for i in range(torch.npu.device_count()):
        d = f"npu:{i}"
        try:
            torch.npu.set_device(d)
            x = torch.randn(64, 64, device=d).half()
            _ = x @ x
            torch.npu.synchronize()
            return d
        except Exception:
            continue
    raise RuntimeError("no healthy NPU available")


def _captured(program, inputs):
    """A zero-arg callable that replays the captured program on `inputs`."""
    gr = GraphReplayExecutor().capture(program, inputs)
    return lambda: gr.replay(inputs, clone=False)


def scan_decision(args):
    B, H, nc = args.B, args.H, args.nc
    dev = args.dev
    inp = make_scan_inputs(B, H, nc, dev)
    ref = scan_reference(inp, B, H, nc)

    staged_prog = build_scan_staged_program(B, H, nc)
    fused_prog = build_scan_fused_program(B, H, nc)

    # Both lowerings must match the fp32 reference before we compare them to
    # each other (the staged lowering is the correctness anchor).
    staged_out = GraphReplayExecutor().capture(staged_prog, inp).replay(inp)
    fused_out = GraphReplayExecutor().capture(fused_prog, inp).replay(inp)
    print("  [scan] vs fp32 reference:")
    for r in gate_outputs(staged_out, ref, tol=2e-2):
        print(f"    staged {r}")
    for r in gate_outputs(fused_out, ref, tol=2e-2):
        print(f"    fused  {r}")

    d = decide("chunk_h_scan", "chunk_h_scan",
               _captured(staged_prog, inp), _captured(fused_prog, inp),
               iters=args.iters)
    return d


def kkt_decision(args):
    nc, H = args.nc, args.H
    inp = make_kkt_inputs(nc, H, args.dev)
    ref = kkt_reference(inp, nc, H)
    fused_prog = build_kkt_fused_program(nc, H)

    gr = GraphReplayExecutor().capture(fused_prog, inp)
    fused_out = gr.replay(inp)
    print("  [kkt] vs fp32 reference:")
    for r in gate_outputs(fused_out, ref, tol=2e-2):
        print(f"    fused  {r}")

    # The kkt gate (coeff = exp(clamp(gv_i - gs_j))) is an outer-difference, not yet
    # expressible as a fuser VecGlueNode, so the staged baseline here is the torch
    # einsum+gate+mask lowering (what the substrate-staged path computes, a chain of
    # dispatched ops round-tripping qk through HBM); fused = the captured kernel.
    d = decide("kkt_gated", "kkt_gated",
               lambda: kkt_reference(inp, nc, H),
               lambda: gr.replay(inp, clone=False),
               iters=args.iters)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--nc", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--stage", choices=["scan", "kkt", "both"], default="both")
    ap.add_argument("--device", default=None, help="npu:N (default: first healthy)")
    args = ap.parse_args()

    args.dev = args.device or healthy_device()
    torch.npu.set_device(args.dev)
    torch.manual_seed(0)
    print(f"device: {args.dev}\n")

    decisions = []
    if args.stage in ("scan", "both"):
        decisions.append(scan_decision(args))
    if args.stage in ("kkt", "both"):
        decisions.append(kkt_decision(args))

    print("\n" + format_fusion_decisions(decisions))


if __name__ == "__main__":
    main()
