"""M3 driver — graph-replay backend (planner lever #1, dispatch-elim).

Captures the staged DeltaNet forward into one NPUGraph and replays it as a single
dispatch, then measures the launch-overhead it removes across a chunk-count sweep
(the "T" axis: fewer chunks = launch-bound, where graph capture wins; more chunks =
compute-bound, where it is perf-neutral). Every captured/replayed output is gated
bit-exact against the staged backend and against the fp32 reference. Also captures
each GDN contraction stage to confirm the backend generalizes to the direct-read
equation family.

    export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
    python run_graph.py --B 2 --H 4 --C 64 --D 128
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

from pto_fuser import (CaptureExecutor, GraphReplayExecutor, gate_outputs)
from pto_fuser.forwards import (build_deltanet_program, deltanet_reference,
                                gdn_contraction_stages, make_inputs)

# Graph replay reruns the *same* kernels, so it must be bit-identical to staged:
# frob_rel is exactly 0.0. The gate is a strict `rel < tol`, so a tiny epsilon
# expresses "bit-exact" (0.0 < BITEXACT passes; any real divergence does not).
BITEXACT = 1e-12


def _time(fn, iters=30, warmup=5):
    """Wall-clock ms/call over `iters` back-to-back calls + ONE trailing sync —
    this is what exposes per-call host dispatch (per-iter sync would hide it)."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = torch.npu.Event(enable_timing=True)
    t1 = torch.npu.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.npu.synchronize()
    return t0.elapsed_time(t1) / iters


def deltanet_sweep(args):
    print(f"\n=== DeltaNet graph-replay  B{args.B} H{args.H} C{args.C} D{args.D} ===")
    dev = torch.device("npu:0")
    scale = args.D ** -0.5
    ok = True
    print(f"  {'nc':>3} {'T':>6} | {'staged':>9} {'graph':>9} | {'speedup':>8} | gate")
    for nc in args.nc:
        prog = build_deltanet_program(args.B, args.H, nc, args.C, args.D, scale)
        binds = make_inputs(args.B, args.H, nc, args.C, args.D, dev)
        ref = deltanet_reference(**binds, B=args.B, H=args.H, nc=nc,
                                 C=args.C, D=args.D, scale=scale)

        staged = CaptureExecutor()                  # persistent-runner eager baseline
        st_out = staged.run(prog, binds)
        gr = GraphReplayExecutor().capture(prog, binds)
        gr_out = gr.replay(binds)

        # correctness: graph == staged (bit-exact) and graph matches fp32 ref
        g1 = all(r.passed for r in gate_outputs(gr_out, st_out, tol=BITEXACT))
        g2 = all(r.passed for r in gate_outputs(gr_out, ref, tol=2e-2))
        ok = ok and g1 and g2

        t_staged = _time(lambda: staged.run(prog, binds))
        t_graph = _time(lambda: gr.replay(binds, clone=False))
        verdict = "OK" if (g1 and g2) else "FAIL"
        print(f"  {nc:>3} {nc*args.C:>6} | {t_staged:>8.3f}m {t_graph:>8.3f}m |"
              f" {t_staged/t_graph:>7.2f}x | {verdict} (eq={int(g1)} ref={int(g2)})")
    return ok


def gdn_stage_capture(args):
    print(f"\n=== GDN stage capture  B{args.B} nc{args.gdn_nc} H{args.H} "
          f"C{args.C} D{args.D} (generality + bit-exact) ===")
    ok = True
    for name, prog, binds in gdn_contraction_stages(
            B=args.B, nc=args.gdn_nc, H=args.H, C=args.C, D=args.D):
        staged = CaptureExecutor()
        st_out = staged.run(prog, binds)
        gr = GraphReplayExecutor().capture(prog, binds)
        gr_out = gr.replay(binds)
        g = all(r.passed for r in gate_outputs(gr_out, st_out, tol=BITEXACT))
        # replay on fresh data still bit-exact vs staged on that data
        binds2 = {k: torch.randn_like(v) for k, v in binds.items()}
        g2 = all(r.passed for r in gate_outputs(
            gr.replay(binds2), staged.run(prog, binds2), tol=BITEXACT))
        ok = ok and g and g2
        print(f"    {name:<9} {prog.nodes[0].equation:<18} "
              f"capture=={'staged' if g else 'FAIL':<6}  replay-newdata="
              f"{'OK' if g2 else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--nc", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--gdn-nc", type=int, default=8)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--D", type=int, default=128)
    args = ap.parse_args()
    torch.npu.set_device("npu:0")
    torch.manual_seed(0)

    ok = deltanet_sweep(args)
    ok = gdn_stage_capture(args) and ok
    print("\n" + ("ALL OK" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
