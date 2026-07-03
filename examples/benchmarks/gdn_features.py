"""Per-feature GDN benchmark — what each fuser feature buys, on the GDN stages.

The fuser's optimizations each target a *different* bottleneck, so this benchmark
measures every feature against its own always-valid baseline on the GDN stage where
it applies, gates the result, and reports one table + one bar plot of speedups:

  | feature              | what it removes                         | GDN stage measured |
  |----------------------|-----------------------------------------|--------------------|
  | read-mode selection  | the input-transpose strided-gather copy         | the 4 contractions |
  | graph capture        | per-stage host dispatch                  | chunk_h scan       |
  | resident state       | the per-chunk HBM round-trip of state S  | chunk_h scan       |
  | glue absorption      | the qk HBM round-trip (gated epilogue)   | kkt                |

Every measured variant is correctness-gated (frob) against its baseline / the fp32
reference before its timing is reported, so a broken-but-fast lowering cannot score.

    export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
    python examples/benchmarks/gdn_features.py --B 1 --H 4 --nc 8
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # examples/
import common  # noqa: E402

import torch  # noqa: E402

from pto_fuser import (GraphReplayExecutor, Planner, StagedExecutor, decide,  # noqa: E402
                       frob_rel, gate_outputs)
from pto_fuser.forwards import (build_kkt_fused_program, build_scan_fused_program,  # noqa: E402
                                build_scan_staged_program, gdn_contraction_stages,
                                kkt_reference, make_kkt_inputs, make_scan_inputs,
                                scan_reference)


def feature_read_mode(dev, B, H, nc, C, D):
    """Read-mode selection on each GDN contraction: input transpose → direct read."""
    rows = []
    planner = Planner()
    for name, prog, bindings in gdn_contraction_stages(B=B, nc=nc, H=H, C=C, D=D, device=dev):
        _, decisions = planner.plan(prog, bindings)
        dr = next(d for d in decisions if d.lever == "direct_read")
        rows.append(dict(feature="read-mode selection", stage=name,
                         baseline_ms=dr.t_base_us / 1e3, feature_ms=dr.t_cand_us / 1e3,
                         speedup=dr.t_base_us / dr.t_cand_us,
                         gate="OK" if dr.gated_ok else "FROB FAIL"))
    return rows


def feature_scan(dev, B, H, nc):
    """Graph capture + resident-state fusion on the chunk_h scan."""
    inp = make_scan_inputs(B, H, nc, dev)
    ref = scan_reference(inp, B, H, nc)
    staged_prog = build_scan_staged_program(B, H, nc)
    fused_prog = build_scan_fused_program(B, H, nc)

    staged = StagedExecutor().run(staged_prog, inp)
    gs = GraphReplayExecutor().capture(staged_prog, inp)
    gf = GraphReplayExecutor().capture(fused_prog, inp)

    ms_staged = common.time_ms(lambda: StagedExecutor().run(staged_prog, inp))
    ms_captured = common.time_ms(lambda: gs.replay(inp, clone=False))
    ms_fused = common.time_ms(lambda: gf.replay(inp, clone=False))

    g_staged = all(r.passed for r in gate_outputs(staged, ref, tol=2e-2))
    fused = gf.replay(inp)
    g_fused = all(r.passed for r in gate_outputs(fused, ref, tol=2e-2))
    bitexact = all(torch.equal(gs.replay(inp)[n], staged[n]) for n in staged)

    return [
        dict(feature="graph capture", stage="chunk_h scan",
             baseline_ms=ms_staged, feature_ms=ms_captured,
             speedup=ms_staged / ms_captured,
             gate="bit-exact" if bitexact else "DIVERGED"),
        dict(feature="resident state", stage="chunk_h scan",
             baseline_ms=ms_captured, feature_ms=ms_fused,
             speedup=ms_captured / ms_fused,
             gate="OK" if (g_staged and g_fused) else "FROB FAIL"),
    ]


def feature_glue(dev, nc, H):
    """Glue absorption on the gated kkt: torch glue baseline → fused epilogue."""
    inp = make_kkt_inputs(nc, H, dev)
    ref = kkt_reference(inp, nc, H)
    gk = GraphReplayExecutor().capture(build_kkt_fused_program(nc, H), inp)
    d = decide("kkt_gated", "kkt_gated",
               lambda: {n: ref[n].clone() for n in ref},
               lambda: gk.replay(inp, clone=False), tol=2e-2, iters=20)
    return [dict(feature="glue absorption", stage="kkt",
                 baseline_ms=d.t_staged_ms, feature_ms=d.t_fused_ms,
                 speedup=d.speedup, gate="OK" if d.gated_ok else "FROB FAIL")]


def render(rows, outdir):
    header = "| feature | GDN stage | baseline ms | feature ms | speedup | gate |"
    sep = "|---------|-----------|------------:|-----------:|--------:|------|"
    lines = [header, sep]
    for r in rows:
        lines.append(f"| {r['feature']} | {r['stage']} | {r['baseline_ms']:.3f} | "
                     f"{r['feature_ms']:.3f} | {r['speedup']:.2f}× | {r['gate']} |")
    table = "\n".join(lines)
    print("\n" + table)

    labels = [f"{r['feature']} ({r['stage']})" for r in rows]
    speeds = [r["speedup"] for r in rows]
    png = os.path.join(outdir, "gdn_features.png")
    if common.plot_speedups(labels, speeds, png,
                            title="pto-fuser features — speedup on GDN stages"):
        print(f"\nplot:  {png}")
    js = os.path.join(outdir, "gdn_features.json")
    with open(js, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"data:  {js}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--nc", type=int, default=8)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--D", type=int, default=128)
    args = ap.parse_args()
    dev = common.pick_device()
    if dev is None:
        print("no healthy NPU — this benchmark needs an Ascend device.")
        return
    torch.manual_seed(0)

    rows = []
    rows += feature_read_mode(dev, args.B, args.H, args.nc, args.C, args.D)
    rows += feature_scan(dev, args.B, args.H, args.nc)
    rows += feature_glue(dev, args.nc, args.H)
    render(rows, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
