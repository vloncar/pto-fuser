"""Graph-replay tests (NPU): the backend captures the staged chain and
replays it as one dispatch — bit-exact, deterministic, and faster in the
launch-bound regime.

Asserts the backend's contract, not a fixed speedup (timings vary by machine):
  * captured replay is bit-identical to the staged backend (same kernels) and
    matches the fp32 reference — for the full DeltaNet forward (incl. the opaque
    tri_inv node, whose stream syncs are dropped under capture) and for each GDN
    contraction stage (the direct-read equation family);
  * replay on FRESH operands (copied into the captured input buffers) still tracks
    a freshly computed reference — capture is reusable, not pinned to one input;
  * two replays are bit-identical (determinism, §6);
  * in the launch-bound regime (small batch) replay is materially faster than the
    staged per-launch baseline — the dispatch-elim win (lever #1).
"""
import pytest
import torch

pytest.importorskip("torch_npu")
import torch_npu  # noqa

from pto_fuser import CaptureExecutor, GraphReplayExecutor, gate_outputs
from pto_fuser.forwards import (build_deltanet_program, deltanet_reference,
                                gdn_contraction_stages, make_inputs)

pytestmark = [pytest.mark.npu, pytest.mark.skipif(
    not torch.npu.is_available(), reason="graph-replay test needs an Ascend NPU")]

BITEXACT = 1e-12  # graph replay reruns the same kernels: frob_rel is exactly 0.0


def _bitexact(got, ref):
    return all(r.passed for r in gate_outputs(got, ref, tol=BITEXACT))


def test_deltanet_capture_bitexact_and_matches_reference():
    torch.npu.set_device("npu:0")
    torch.manual_seed(0)
    dev = torch.device("npu:0")
    B, H, nc, C, D = 2, 4, 4, 64, 128
    scale = D ** -0.5

    prog = build_deltanet_program(B, H, nc, C, D, scale)
    binds = make_inputs(B, H, nc, C, D, dev)
    ref = deltanet_reference(**binds, B=B, H=H, nc=nc, C=C, D=D, scale=scale)

    staged = CaptureExecutor().run(prog, binds)
    gr = GraphReplayExecutor().capture(prog, binds)
    got = gr.replay(binds)

    assert _bitexact(got, staged), "graph replay != staged backend (same kernels)"
    bad = [str(r) for r in gate_outputs(got, ref, tol=2e-2) if not r.passed]
    assert not bad, "captured DeltaNet diverged from fp32 reference:\n" + "\n".join(bad)


def test_replay_on_fresh_data_tracks_reference():
    torch.npu.set_device("npu:0")
    torch.manual_seed(1)
    dev = torch.device("npu:0")
    B, H, nc, C, D = 2, 4, 3, 64, 128
    scale = D ** -0.5

    prog = build_deltanet_program(B, H, nc, C, D, scale)
    gr = GraphReplayExecutor().capture(prog, make_inputs(B, H, nc, C, D, dev))

    binds2 = make_inputs(B, H, nc, C, D, dev)            # different operands
    got2 = gr.replay(binds2)
    ref2 = deltanet_reference(**binds2, B=B, H=H, nc=nc, C=C, D=D, scale=scale)
    bad = [str(r) for r in gate_outputs(got2, ref2, tol=2e-2) if not r.passed]
    assert not bad, "replay on fresh data diverged from its reference:\n" + "\n".join(bad)


def test_replay_is_deterministic():
    torch.npu.set_device("npu:0")
    torch.manual_seed(2)
    dev = torch.device("npu:0")
    B, H, nc, C, D = 2, 4, 4, 64, 128
    prog = build_deltanet_program(B, H, nc, C, D, D ** -0.5)
    binds = make_inputs(B, H, nc, C, D, dev)
    gr = GraphReplayExecutor().capture(prog, binds)
    a, b = gr.replay(binds), gr.replay(binds)
    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} not bit-identical across replays"


def test_replay_shape_mismatch_raises():
    torch.npu.set_device("npu:0")
    dev = torch.device("npu:0")
    B, H, nc, C, D = 2, 4, 2, 64, 128
    prog = build_deltanet_program(B, H, nc, C, D, D ** -0.5)
    gr = GraphReplayExecutor().capture(prog, make_inputs(B, H, nc, C, D, dev))
    wrong = make_inputs(B, H, nc + 1, C, D, dev)         # different chunk count
    with pytest.raises(ValueError):
        gr.replay(wrong)


def test_gdn_stages_capture_bitexact():
    torch.npu.set_device("npu:0")
    torch.manual_seed(0)
    for name, prog, binds in gdn_contraction_stages(B=1, nc=4, H=16, C=64, D=128):
        staged = CaptureExecutor()
        gr = GraphReplayExecutor().capture(prog, binds)
        assert _bitexact(gr.replay(binds), staged.run(prog, binds)), \
            f"{name}: graph replay != staged"
        fresh = {k: torch.randn_like(v) for k, v in binds.items()}
        assert _bitexact(gr.replay(fresh), staged.run(prog, fresh)), \
            f"{name}: replay on fresh data != staged"


def test_dispatch_elim_win_in_launch_bound_regime():
    torch.npu.set_device("npu:0")
    torch.manual_seed(0)
    dev = torch.device("npu:0")
    B, H, nc, C, D = 2, 4, 4, 64, 128               # small batch -> launch-bound
    prog = build_deltanet_program(B, H, nc, C, D, D ** -0.5)
    binds = make_inputs(B, H, nc, C, D, dev)
    staged = CaptureExecutor()
    gr = GraphReplayExecutor().capture(prog, binds)

    def _time(fn, iters=30):
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        t0, t1 = (torch.npu.Event(enable_timing=True) for _ in range(2))
        t0.record()
        for _ in range(iters):
            fn()
        t1.record()
        torch.npu.synchronize()
        return t0.elapsed_time(t1) / iters

    t_staged = _time(lambda: staged.run(prog, binds))
    t_graph = _time(lambda: gr.replay(binds, clone=False))
    # measured ~2.5-4x; assert a conservative margin to stay robust to jitter.
    assert t_staged / t_graph > 1.3, f"no dispatch-elim win: {t_staged:.3f} vs {t_graph:.3f}ms"
