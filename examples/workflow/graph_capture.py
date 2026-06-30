"""Workflow demo: graph capture (dispatch elimination).

Wrap the staged DeltaNet forward in a single NPUGraph capture and replay it as one
dispatch. Replay is bit-exact vs the staged backend; the win is regime-specific —
a real multiplier when host launch dominates (small batch / few large chunks),
perf-neutral once device work per launch hides the dispatch.

    python examples/workflow/graph_capture.py --B 2 --H 4 --nc 1 2 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # examples/
import common  # noqa: E402

import torch  # noqa: E402

from pto_fuser import (GraphReplayExecutor, StagedExecutor, frob_rel,  # noqa: E402
                       gate_outputs)
from pto_fuser.forwards import (build_deltanet_program, deltanet_reference,  # noqa: E402
                                make_inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--nc", type=int, nargs="+", default=[1, 2, 8])
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--D", type=int, default=128)
    args = ap.parse_args()
    dev = common.pick_device()
    if dev is None:
        print("no healthy NPU — this demo needs an Ascend device.")
        return
    torch.manual_seed(0)
    scale = args.D ** -0.5

    rows, labels, speedups = [], [], []
    for nc in args.nc:
        prog = build_deltanet_program(args.B, args.H, nc, args.C, args.D, scale)
        inp = make_inputs(args.B, args.H, nc, args.C, args.D, dev)
        staged = StagedExecutor().run(prog, inp)
        gr = GraphReplayExecutor().capture(prog, inp)
        replayed = gr.replay(inp)

        ref = deltanet_reference(**inp, B=args.B, H=args.H, nc=nc, C=args.C, D=args.D, scale=scale)
        bad = [str(r) for r in gate_outputs(replayed, ref, tol=2e-2) if not r.passed]
        bitexact = all(torch.equal(replayed[n], staged[n]) for n in staged)
        ms_staged = common.time_ms(lambda: StagedExecutor().run(prog, inp))
        ms_graph = common.time_ms(lambda: gr.replay(inp, clone=False))
        T = nc * args.C
        note = ("bit-exact" if bitexact else f"DIVERGED max={max(frob_rel(replayed[n], staged[n]) for n in staged):.1e}")
        note += "" if not bad else " | GATE FAIL"
        rows.append(common.Measurement(f"nc={nc} (T={T}) staged", ms_staged))
        m = common.Measurement(f"nc={nc} (T={T}) graph", ms_graph, note=note)
        m.relative_to(ms_staged)
        rows.append(m)
        labels.append(f"nc={nc} (T={T})")
        speedups.append(ms_staged / ms_graph)

    print(common.format_table(rows, title="DeltaNet — staged vs graph-replay"))
    png = os.path.join(os.path.dirname(__file__), "graph_capture_speedup.png")
    if common.plot_speedups(labels, speedups, png, title="Graph capture: dispatch elimination"):
        print(f"\nplot: {png}")


if __name__ == "__main__":
    main()
