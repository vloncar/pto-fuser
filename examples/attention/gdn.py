"""Gated DeltaNet (GDN) as a pto-fuser forward.

GDN is the *delta-rule* family (not the plain linear recurrence): each chunk solves a
small triangular system (the WY representation) before the cross-chunk scan. The
backbone is exactly the DeltaNet forward shipped in ``pto_fuser.forwards`` —

    kkt → (I+A)⁻¹  → recompute W,U → cross-chunk scan → chunk output

— and GDN adds gating: an ``exp(g)`` decay on the keys/state. Two stages carry that
gating and are the ones worth *fusing*, so this example shows both the staged
backbone forward and the per-stage fusion decision on the two GDN-characteristic
fused kernels:

  * ``kkt_gated``     — the kkt contraction with the gated + causal-masked epilogue
                        folded into the matmul store (the qk matrix never lands in HBM);
  * ``chunk_h_scan``  — the cross-chunk recurrence with the decaying state kept
                        resident across chunks instead of round-tripping HBM.

    python examples/attention/gdn.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from common import format_table, pick_device, time_ms  # noqa: E402
from pto_fuser import (GraphReplayExecutor, StagedExecutor, decide,  # noqa: E402
                       gate_outputs)
from pto_fuser.forwards import (build_deltanet_program, build_kkt_fused_program,
                                build_scan_fused_program, build_scan_staged_program,
                                deltanet_reference, kkt_reference, make_inputs,
                                make_kkt_inputs, make_scan_inputs, scan_reference)


def backbone(dev, B=2, H=4, nc=4, C=64, D=128):
    """The delta-rule backbone: run the DeltaNet forward staged, gate vs fp32."""
    print(f"\n[backbone] DeltaNet/GDN forward  B={B} H={H} nc={nc} C={C} D={D}")
    scale = D ** -0.5
    inp = make_inputs(B, H, nc, C, D, dev)
    prog = build_deltanet_program(B, H, nc, C, D, scale)
    ref = deltanet_reference(inp["q"], inp["k"], inp["v"], inp["beta"], B, H, nc, C, D, scale)
    got = StagedExecutor().run(prog, inp)
    bad = [str(r) for r in gate_outputs(got, ref, tol=2e-2) if not r.passed]
    print("  staged forward:", "ALL OK" if not bad else "FAIL\n   " + "\n   ".join(bad))


def fused_stage_decisions(dev, B=1, H=4, nc=8):
    """The two GDN-characteristic fused stages, each as a gated staged-vs-fused
    decision (frob ≡ staged + determinism + faster)."""
    print(f"\n[fusion] GDN gated stages  B={B} H={H} nc={nc}")

    # resident-state scan (decaying state kept on-chip across chunks)
    scan_in = make_scan_inputs(B, H, nc, dev)
    gs = GraphReplayExecutor().capture(build_scan_staged_program(B, H, nc), scan_in)
    gf = GraphReplayExecutor().capture(build_scan_fused_program(B, H, nc), scan_in)
    d_scan = decide("chunk_h_scan", "chunk_h_scan",
                    lambda: gs.replay(scan_in, clone=False),
                    lambda: gf.replay(scan_in, clone=False), tol=2e-2, iters=20)
    print("  " + str(d_scan))

    # gated kkt (qk + gated/masked epilogue fused; staged baseline = torch reference)
    kkt_in = make_kkt_inputs(nc, H, dev)
    kref = kkt_reference(kkt_in, nc, H)
    gk = GraphReplayExecutor().capture(build_kkt_fused_program(nc, H), kkt_in)
    d_kkt = decide("kkt_gated", "kkt_gated",
                   lambda: {n: kref[n].clone() for n in kref},
                   lambda: gk.replay(kkt_in, clone=False), tol=2e-2, iters=20)
    print("  " + str(d_kkt))


def main():
    dev = pick_device()
    if dev is None:
        print("no healthy NPU — building the GDN Program off-NPU to check it constructs.")
        build_deltanet_program(2, 4, 4, 64, 128, 128 ** -0.5)
        print("Program built OK.")
        return
    torch.manual_seed(0)
    backbone(dev)
    fused_stage_decisions(dev)


if __name__ == "__main__":
    main()
