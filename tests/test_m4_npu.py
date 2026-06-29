"""M4 exit test (NPU): the fused-node backend + fusion decision (levers 5/6).

Asserts the backend's contract, not a fixed speedup (timings vary, esp. on a shared
box):
  * the fused resident-state scan (lever 5) and the fused gated kkt (lever 6) each
    match the fp32 reference, and the scan's fused output is **bit-identical** to the
    staged lowering it replaces (same numerics, one kernel vs many);
  * both fused kernels are deterministic (run twice, bit-identical — design §6,
    mandatory on any fused lowering);
  * a fused node captures + replays through the M3 graph backend (single dispatch),
    bit-exact;
  * `fusion.decide` keeps the scan fusion (it gates green, is deterministic, and is
    faster than staged-captured — the resident state removes the per-chunk HBM
    round-trip).
"""
import pytest
import torch

pytest.importorskip("torch_npu")
import torch_npu  # noqa

from conftest import healthy_npu
from pto_fuser import (StagedExecutor, GraphReplayExecutor, decide, frob_rel,
                       gate_determinism, gate_outputs)
from pto_fuser.forwards import (build_kkt_fused_program, build_scan_fused_program,
                                build_scan_staged_program, kkt_reference,
                                make_kkt_inputs, make_scan_inputs, scan_reference)

DEV = healthy_npu()
pytestmark = pytest.mark.skipif(DEV is None, reason="M4 fused-node test needs a healthy Ascend NPU")
TOL = 2e-2


def _bitexact(got, ref):
    return all(torch.equal(got[n], ref[n]) for n in ref if n in got)


def test_fused_scan_matches_reference_and_staged():
    torch.npu.set_device(DEV)
    torch.manual_seed(0)
    B, H, nc = 1, 2, 3
    inp = make_scan_inputs(B, H, nc, DEV)
    ref = scan_reference(inp, B, H, nc)
    ex = StagedExecutor()

    fused = ex.run(build_scan_fused_program(B, H, nc), inp)
    staged = ex.run(build_scan_staged_program(B, H, nc), inp)

    bad = [str(r) for r in gate_outputs(fused, ref, tol=TOL) if not r.passed]
    assert not bad, "fused scan diverged from fp32 reference:\n" + "\n".join(bad)
    # staged and fused compute the SAME recurrence; here they come out bit-identical.
    assert _bitexact(fused, staged), "fused scan != staged lowering"


def test_fused_kkt_matches_reference():
    torch.npu.set_device(DEV)
    torch.manual_seed(0)
    nc, H = 3, 4
    inp = make_kkt_inputs(nc, H, DEV)
    ref = kkt_reference(inp, nc, H)
    got = StagedExecutor().run(build_kkt_fused_program(nc, H), inp)
    bad = [str(r) for r in gate_outputs(got, ref, tol=TOL) if not r.passed]
    assert not bad, "fused kkt diverged from fp32 reference:\n" + "\n".join(bad)


def test_fused_kernels_deterministic():
    torch.npu.set_device(DEV)
    torch.manual_seed(1)
    B, H, nc = 1, 2, 3
    inp = make_scan_inputs(B, H, nc, DEV)
    ex = StagedExecutor()
    prog = build_scan_fused_program(B, H, nc)
    assert gate_determinism(lambda: ex.run(prog, inp)).passed, "scan fused NDET"
    ki = make_kkt_inputs(nc, H, DEV)
    kprog = build_kkt_fused_program(nc, H)
    assert gate_determinism(lambda: ex.run(kprog, ki)).passed, "kkt fused NDET"


def test_fused_node_captures_and_replays_bitexact():
    torch.npu.set_device(DEV)
    torch.manual_seed(2)
    B, H, nc = 1, 2, 4
    inp = make_scan_inputs(B, H, nc, DEV)
    eager = StagedExecutor().run(build_scan_fused_program(B, H, nc), inp)
    gr = GraphReplayExecutor().capture(build_scan_fused_program(B, H, nc), inp)
    replayed = gr.replay(inp)
    assert _bitexact(replayed, eager), "captured fused node != eager"
    # replay is itself deterministic
    assert _bitexact(gr.replay(inp), replayed)


def test_decide_keeps_scan_fusion():
    torch.npu.set_device(DEV)
    torch.manual_seed(0)
    B, H, nc = 1, 4, 8                 # launch-bound: resident state clearly wins
    inp = make_scan_inputs(B, H, nc, DEV)
    staged_prog = build_scan_staged_program(B, H, nc)
    fused_prog = build_scan_fused_program(B, H, nc)
    gs = GraphReplayExecutor().capture(staged_prog, inp)
    gf = GraphReplayExecutor().capture(fused_prog, inp)
    d = decide("chunk_h_scan", "chunk_h_scan",
               lambda: gs.replay(inp, clone=False),
               lambda: gf.replay(inp, clone=False), tol=TOL, iters=20)
    assert d.gated_ok and d.deterministic, str(d)
    assert d.kept and d.faster, f"resident-state scan not kept: {d}"
