"""Fusion tests (NPU): the fused-node backend + the staged-vs-fused decision.

Asserts the backend's contract, not a fixed speedup (timings vary, esp. on a shared
box):
  * the fused resident-state scan and the fused gated kkt each
    match the fp32 reference, and the scan's fused output is **bit-identical** to the
    staged lowering it replaces (same numerics, one kernel vs many);
  * both fused kernels are deterministic (run twice, bit-identical — design §6,
    mandatory on any fused lowering);
  * a fused node captures + replays through the graph-replay backend (single dispatch),
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

pytestmark = pytest.mark.npu
DEV = None
TOL = 2e-2


@pytest.fixture(autouse=True)
def _device():
    """Resolve a healthy NPU lazily (at test time, not collection time — probing a
    wedged chip at import would hang the whole suite on a shared box)."""
    global DEV
    if DEV is None:
        DEV = healthy_npu()
    if DEV is None:
        pytest.skip("no healthy Ascend NPU")
    torch.npu.set_device(DEV)


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


def test_qkv_flash_matches_reference_and_v1_equals_v2():
    """The B3 flash kernel (chunk_o q·kᵀ → gate/mask → ·v in one launch) matches the
    fp32 staged reference for both variants, and the interleaved V2 (S/A L2-resident)
    is bit-identical to the two-pass V1 — the 3-stage Cube-Vec-Cube FFTS choreography
    is exact. Deterministic on each variant."""
    from pto_fuser.fused import shared_fused_registry
    torch.npu.set_device(DEV)
    nc, H, C, D = 4, 4, 128, 128
    M = nc * H
    g = torch.Generator(device="cpu").manual_seed(3)
    q = (torch.randn(M, C, D, generator=g) * 0.1).half().to(DEV)
    k = (torch.randn(M, C, D, generator=g) * 0.1).half().to(DEV)
    v = (torch.randn(M, C, D, generator=g) * 0.1).half().to(DEV)
    gate = (torch.randn(M, C, generator=g) * 0.1).float().to(DEV)
    beta1 = torch.ones(M, C, dtype=torch.float16, device=DEV)
    params = {"nc": nc, "H": H, "C": C, "D": D, "DV": D, "causal": True}
    reg = shared_fused_registry()

    rows = torch.arange(C, device=DEV)
    mask = (rows[:, None] >= rows[None, :]).float()
    S = q.float() @ k.float().transpose(-1, -2)
    coef = torch.exp(torch.clamp(gate[:, :, None] - gate[:, None, :], max=0.0))
    ref = ((S * coef * mask).half().float()) @ v.float()

    (o1,) = reg.run("qkv_flash_native", [q, k, v, gate, beta1], params)
    (o2,) = reg.run("qkv_flash_native_v2", [q, k, v, gate, beta1], params)
    assert frob_rel(o1, ref) < TOL and frob_rel(o2, ref) < TOL, "flash != reference"
    assert torch.equal(o1, o2), "flash V2 (interleaved) != V1 (two-pass)"
    assert gate_determinism(lambda: {"o": reg.run("qkv_flash_native", [q, k, v, gate, beta1], params)[0]}).passed
    assert gate_determinism(lambda: {"o": reg.run("qkv_flash_native_v2", [q, k, v, gate, beta1], params)[0]}).passed


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
