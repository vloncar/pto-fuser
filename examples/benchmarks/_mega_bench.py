"""Shared driver for the end-to-end *pto-fuser vs megakernel* benchmarks (GDN, KDA).

This is the full equivalent of `pto-einsum/benchmarks/complex/{gdn,kda}/bench_*_4way.py`
done through the fusion layer: the same gated forward, but built as a fuser `Program`
and executed by the fuser, put head-to-head against the megagdn / megakda megakernel
across a grid of head counts and sequence lengths.

Two implementations per config, **megakernel is the baseline (1.00×)**:

  * ``megagdn`` / ``megakda`` — the hand-written fused megakernel (`run_full_pipeline`)
  * ``pto-fuser``             — the same forward as a fuser `Program`, executed as one
                                `NPUGraph` (graph-captured, dispatch-eliminated)

Both are correctness-gated (Frobenius relative error) against the fp32 reference before
any timing is reported. The captured fuser forward is additionally checked bit-exact
against a one-shot staged run of the same `Program` (an untimed correctness cross-check,
not a reported row). megagdn is run twice and the consistent (min) error is reported,
flagging any nondeterminism — exactly as the 4-way bench does.

A "family" object supplies the forward (see ``gdn_mega.py`` / ``kda_mega.py``):
``make_inputs, reference, build, prepare`` (fuser side) and ``mega_runner,
to_mega_golden`` (megakernel side).
"""
from __future__ import annotations

import json
import os

import torch

import common
from common import Measurement, format_table
from pto_fuser import (Features, GraphReplayExecutor, StagedExecutor,
                       compile_program, frob_rel)


def _lowering(prog, binds, ref):
    """Capture one Program; return (replay-runner, captured-frob-vs-ref, capture-faithful)."""
    staged = StagedExecutor().run(prog, binds)["o"]
    gr = GraphReplayExecutor().capture(prog, binds)
    cap = gr.replay(binds)["o"]
    return gr, frob_rel(cap, ref), bool(torch.equal(cap, staged))


def _measure_config(fam, dev, H, nc, iters, mega_name):
    B, C, D = 1, 128, 128
    T = nc * C
    scale = D ** -0.5
    torch.manual_seed(0)

    inp = fam.make_inputs(B, H, nc, C, D, dev)
    ref = fam.reference(inp, scale)                       # [B,H,nc,C,D] fp32 golden
    binds = fam.prepare(inp)

    # fuser: canonical forward -> compile (propose/verify/dispose over the transform
    # pipeline: resident-state scan, glue absorption, read-mode/fused-store selection,
    # each kept only on a gated + deterministic + measured win vs the canonical floor).
    canon = fam.build(B, H, nc, C, D, scale)
    result = compile_program(canon, Features(B, H, nc, C, D), bindings=binds,
                             verify=True, iters=iters)
    gr, f_fuser, faithful = _lowering(result.program, binds, ref)
    kept = [r.name for r in result.report.kept]
    compile_note = "  [compiled: " + (", ".join(kept) if kept else "canonical") + "]"

    # megakernel (2 reps; report consistent error, flag nondeterminism)
    mega = fam.mega_runner(inp, B, H, nc, C, D, scale, dev)
    golden_m = fam.to_mega_golden(ref)
    reps = [frob_rel(mega(), golden_m) for _ in range(2)]
    f_mega = min(reps)
    ndet = max(reps) > 2 * min(reps) + 1e-3

    t_fuser = common.time_ms(lambda: gr.replay(binds, clone=False), iters=iters)
    t_mega = common.time_ms(mega, iters=iters)

    cap_note = "capture bit-exact" if faithful else "capture DIVERGED from staged"
    mega_note = f"frob {f_mega:.1e}" + ("  [NONDET]" if ndet else "")
    rows = [
        Measurement(mega_name, t_mega, note=mega_note),
        Measurement("pto-fuser", t_fuser,
                    note=f"{len(kept)} transforms; frob {f_fuser:.1e}; {cap_note}"),
    ]
    return rows, compile_note, dict(H=H, nc=nc, T=T, transforms_kept=kept,
                                    frob=dict(fuser=f_fuser, mega=f_mega),
                                    ms=dict(fuser=t_fuser, mega=t_mega),
                                    speedup_vs_mega=t_mega / t_fuser,
                                    bitexact=bool(faithful), mega_nondet=bool(ndet))


def run(fam, dev, configs, iters, outdir, title, slug, mega_name="megagdn"):
    print(f"{title}\nconfigs (H×nc, B=1, C=D=128): {configs}   iters={iters}")
    print(f"baseline = {mega_name} (1.00×); pto-fuser = graph-captured forward")
    print("=" * 70)
    blob = {"configs": [list(c) for c in configs], "iters": iters,
            "mega": mega_name, "baseline": mega_name, "results": {}}
    plot_labels, plot_speeds = [], []
    for (H, nc) in configs:
        rows, scan_note, rec = _measure_config(fam, dev, H, nc, iters, mega_name)
        tag = f"H{H}·nc{nc}·T{rec['T']}"
        print("\n" + format_table(rows, title=tag, baseline=mega_name))
        if scan_note:
            print(scan_note.strip())
        blob["results"][f"{H}x{nc}"] = rec
        plot_labels.append(tag)
        plot_speeds.append(rec["speedup_vs_mega"])     # pto-fuser ms-speedup over mega

    js = os.path.join(outdir, f"{slug}_results.json")
    with open(js, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\ndata:  {js}")
    png = os.path.join(outdir, f"{slug}.png")
    if common.plot_speedups(plot_labels, plot_speeds, png,
                            title=f"{title} — pto-fuser speedup vs {mega_name} (1.0 = parity)"):
        print(f"plot:  {png}")


def parse_configs(s, default):
    if not s:
        return default
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        h, nc = tok.lower().split("x")
        out.append((int(h), int(nc)))
    return out
