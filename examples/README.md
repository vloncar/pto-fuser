# pto-fuser examples

Runnable demonstrations of the fusion layer, from a one-screen "hello world" up to a
zoo of chunked-attention mechanisms expressed on the IR. Everything runs from a
checkout — no install — and degrades gracefully off-NPU: the `Program`s still build, so
you can read the construction (and the off-NPU test suite exercises that path) even
without a device. The numeric runs and benchmarks need a healthy Ascend NPU.

## Setup

```bash
export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
python examples/minimal.py
```

Device selection is automatic — `common.device.pick_device` reads `npu-smi info` and
picks a chip that is Health `OK` with no running process (the box is shared), so you
never have to choose `npu:N` by hand. If no chip is usable a script prints a short
"no healthy NPU" note and (where it can) still builds the `Program` so the run isn't a
hard error.

## Layout

```
examples/
├── common/                 # shared infra the examples reuse
│   ├── device.py           #   healthy/idle-NPU selection via `npu-smi info`
│   ├── bench.py            #   ms/call timing (dispatch-exposing) + markdown tables
│   └── plot.py             #   speedup bar charts / sweep line plots (matplotlib, optional)
├── minimal.py              # the smallest program: one contraction + a masked, scaled epilogue
├── attention/              # the advanced examples — chunked-attention mechanisms
│   ├── _chunked.py         #   shared chunked-linear-attention core (builder + fp32 reference)
│   ├── __init__.py         #   gate generators + run_linear_variant driver
│   ├── vanilla_la.py       #   plain linear attention   (no decay)
│   ├── retnet.py           #   RetNet retention         (per-head scalar decay)
│   ├── mamba2.py           #   Mamba-2 / SSD            (per-token scalar decay)
│   ├── gla.py              #   gated linear attention   (per-channel decay)
│   ├── gdn.py              #   gated DeltaNet           (delta-rule + gating, fused stages)
│   ├── kda.py              #   Kimi delta attention     (GDN + per-dim gate; same graph)
│   ├── _gdn_full.py        #   the COMPLETE gated GDN forward as one Program (+ fp32 ref)
│   └── _kda_full.py        #   the COMPLETE per-dim-gated KDA forward as one Program
├── workflow/               # one demo per fuser feature
│   ├── read_modes.py       #   read-mode / fused-output selection (the Planner)
│   ├── graph_capture.py    #   dispatch elimination via NPUGraph capture
│   └── fusion_decision.py  #   staged-vs-fused decision (gate + measure)
└── benchmarks/
    ├── gdn_features.py     # what each feature buys, measured per GDN stage (table + plot)
    ├── gdn_mega.py         # end-to-end: fuser GDN forward (staged/captured) vs megagdn
    ├── kda_mega.py         # end-to-end: fuser KDA forward (staged/captured) vs megakda
    └── _mega_bench.py      #   shared driver for the two vs-megakernel benchmarks
```

## The examples

### `minimal.py` — start here
The smallest complete program: one einsum contraction (`ntd,nsd->nts`, scaled-dot
attention scores) followed by a causal mask and a scale, built directly from the IR
node types. Runs it staged and prints `frob_rel` against `torch`. No flags. This is the
"how do I build and run a Program" template.

```bash
python examples/minimal.py
```

### `attention/` — chunked linear attention (vanilla LA · RetNet · Mamba-2 · GLA)
These four are **the same chunked recurrence with a different per-token gate**, so they
share one builder, [`_chunked.py`](attention/_chunked.py), and differ by a few lines.
The chunk decomposition is: within a chunk, take the inclusive cumulative gate `P`;
form `q̃ = q⊙P`, `k̂ = k/P`, `k̄ = k⊙(γ/P)` (with `γ` the chunk's total decay); then
`O = tril(q̃k̂ᵀ,0)·V + q̃·S_in` and `S_out = diag(γ)·S_in + k̄ᵀ·V`. The `Program` is
einsum cores + Vec glue (the masks/decays) + a cross-chunk scan over `S`; the gate is
the *only* thing that changes between mechanisms:

| script         | mechanism             | gate                          |
|----------------|-----------------------|-------------------------------|
| `vanilla_la.py`| linear attention      | `g = 1` (no decay)            |
| `retnet.py`    | RetNet retention      | per-head scalar `γ_h`         |
| `mamba2.py`    | Mamba-2 / SSD         | per-token scalar `a_t`        |
| `gla.py`       | gated linear attention| per-channel vector            |

Each script is a few lines: pick a gate generator and hand it to `run_linear_variant`
(in [`attention/__init__.py`](attention/__init__.py)), which builds the Program, runs it
staged, checks the output against a **token-recurrent fp32 reference** (the definition
the chunk form must reproduce), confirms graph capture/replay is bit-exact, and prints a
timing table. No flags:

```bash
python examples/attention/vanilla_la.py
python examples/attention/retnet.py
python examples/attention/mamba2.py
python examples/attention/gla.py
```

### `attention/gdn.py`, `attention/kda.py` — delta rule
GDN and KDA aren't a single contraction: each chunk first solves a small triangular
system (the WY representation) before the scan, so these reuse the DeltaNet forward in
[`pto_fuser.forwards`](../src/pto_fuser/forwards) rather than `_chunked.py`. `gdn.py` has
two parts: `backbone()` runs the staged DeltaNet/GDN forward and checks every stage, and
`fused_stage_decisions()` runs the `decide()` gate on the two stages worth fusing
(`chunk_h_scan`, `kkt_gated`). `kda.py` reuses that machinery — KDA is GDN with a per-dim
gate baked into the operands, i.e. the *same* einsum graph. Both take `--B --H --nc --C
--D`:

```bash
python examples/attention/gdn.py --B 1 --H 4 --nc 8
python examples/attention/kda.py --B 1 --H 4 --nc 8
```

### `workflow/` — one demo per fuser feature
Each shows a single optimization feature in isolation on the DeltaNet/GDN forward:

```bash
python examples/workflow/read_modes.py     --B 8 --H 32 --nc 8     # Planner: read-mode + fused-output selection
python examples/workflow/graph_capture.py  --B 2 --H 4 --nc 1 2 8  # dispatch elim via NPUGraph (sweeps nc)
python examples/workflow/fusion_decision.py --B 1 --H 4 --nc 8     # staged-vs-fused gate+measure
```

`read_modes.py` prints the Planner's per-stage decisions and the glue-absorption
candidates; `graph_capture.py` sweeps the chunk count and prints a staged-vs-graph-replay
table (and a plot); `fusion_decision.py` prints the keep/reject decision for the scan and
kkt kernels.

## Benchmarking

`benchmarks/gdn_features.py` answers "what did each feature buy?" — it measures the
features against their representative GDN stage and emits a markdown table, a bar plot,
and a JSON dump:

```bash
python examples/benchmarks/gdn_features.py --B 1 --H 4 --nc 8 --C 64 --D 128
```

It covers the read-mode selection (Planner, base-vs-candidate µs per stage), graph
capture + resident state (on the `chunk_h` scan), and glue absorption (on `kkt`). Every
candidate is correctness-gated before its speedup is reported, so a number only appears
if it reproduced the staged result.

The timing primitive is `common.bench.time_ms` — back-to-back launches with a single
trailing synchronize, so dispatch/launch overhead stays *in* the measurement (that
overhead is exactly what graph capture removes, so hiding it would defeat the point). The
plots come from `common.plot` (`plot_speedups`, `plot_sweep`); matplotlib is imported
lazily, so if it isn't installed the tables/JSON still print and only the PNG is skipped.

`run_linear_variant` (the `attention/*` scripts) also prints a small per-variant timing
table using the same infra, so the linear examples double as quick benchmarks.

### End-to-end vs. the megakernel

`benchmarks/gdn_mega.py` and `benchmarks/kda_mega.py` run the **complete** gated GDN / KDA
forward — built as one fuser `Program` in `attention/_gdn_full.py` / `_kda_full.py` — and
put it head-to-head with the hand-written megakernel (`megagdn` / `megakda`), end-to-end,
across a grid of head counts and sequence lengths:

```bash
python examples/benchmarks/gdn_mega.py --configs 16x4,16x8,16x16,32x4,32x8,64x4,64x8
python examples/benchmarks/kda_mega.py --configs 16x4,16x8,16x16,32x4,32x8,64x4,64x8
```

Two implementations per config, with the **megakernel as the baseline (1.00×)**: `megagdn`
/ `megakda`, and `pto-fuser` — the same `Program` executed as one `NPUGraph` (graph-captured,
so the fuser side is a single dispatch, the fair analogue of a megakernel). Each `HxNC` token
is a head count `H` and a chunk count `nc`, giving sequence length `T = nc·128`; the table is
read across heads and sequence lengths. Both are Frobenius-gated against the fp32 reference,
and the captured fuser forward is additionally checked bit-exact against an untimed staged run
of the same `Program`.

Both GDN and KDA offer the **`chunk_h_scan` resident-state lowering** on the `pto-fuser` row:
the cross-chunk recurrence's carried state `S` is `O(H·D²)`, and the default einsum scan
round-trips it through HBM every chunk — bandwidth that graph capture can't remove and that
grows with head count and sequence length, which is exactly where the megakernel pulls ahead.
The fused lowering keeps `S` on-chip across chunks (the gated recurrence is mapped onto the
kernel by absorbing the per-token gate into `k` and feeding the decay separately; `v_new` is
recovered as one parallel batched matmul). GDN's decay is a single scalar `exp(gₗ)` per chunk;
**KDA's is a per-dimension `[D]` vector** `exp(g_tot)` (one rate per `K`-row of `S`), which the
kernel applies with a single fused `TROWEXPANDMUL` (the decay vector expanded down the columns
and multiplied into `S` in one instruction — no materialized broadcast tile) under a
`SCAN_PERDIM_DECAY` build flag (a distinct `.so`; GDN's stays byte-identical). It is wired in **behind the fusion-decide
gate** — kept only when it gates bit-faithful to the einsum scan, runs deterministically, *and*
measures faster — and each config prints the verdict (`[scan FUSE: forward …→… ms (…×) …]`).
For GDN the configs that lost to the megakernel (large `H`, long scans) move to
parity-or-better. For KDA the fused scan is kept on every config and wins at launch-bound shapes
(H16·nc4 ≈ 1.7×); the fused `TROWEXPANDMUL` keeps the per-dim decay cheap enough that the
largest configs (H32·nc8, H64·nc4) reach roughly parity with the megakernel end-to-end (the
in-process fused-vs-einsum scan stage is a steady 1.07–1.30×; the vs-megakernel ratio itself
is noisy on a shared device). The whole gated pipeline lives on the IR: einsum cores +
`mul/sub/add/tril/scale` glue + the opaque triangular inverse, with the cumulative-sum / `exp`
gate arithmetic precomputed host-side into coefficient tensors (the same move the linear
examples make for decay) — and the host-coefficient decomposition is proven equal to the
`RefGDN` / `RefKDA` oracle to fp32 precision. The takeaway it surfaces: the graph-captured
fuser forward matches the megakernel at launch-bound shapes (small `H`, short sequences) and
the megakernel pulls ahead as the per-chunk work and scan length grow — its resident state
avoids the per-chunk HBM round-trips the staged scan pays.

Simplification: these use `Hg = H` (no GQA); the megakernels are run with `key_heads = H`
to match. The fuser forward is head-agnostic, but the megakernel's compiled dispatch only
supports `H ∈ {16, 24, 32, 48, 64}`, so the comparison configs use those head counts.

## Relation to the pto-einsum benchmarks

`benchmarks/{gdn,kda}_mega.py` are the **fusion-layer equivalent** of
`pto-einsum/benchmarks/complex/{gdn,kda}/bench_*_4way.py`: same gated forward, same
megakernel column, same correctness gating — but where the pto-einsum 4-way compares the
**contraction backends** (torch reference · torch.einsum · pto-einsum · the megakernel
fork) stage-by-stage, these compare the **whole fuser forward** (graph-captured) directly
against that same megakernel, end-to-end. Keep the 4-way in pto-einsum as the
contraction-backend showcase; the end-to-end fusion story lives here. The `gdn_features.py` benchmark sits between them — it attributes the fuser's
end-to-end win to the individual features (read mode, graph capture, resident state, glue
absorption) on the representative GDN stages.

## Adding a mechanism

A linear-recurrence variant is just a gate function — see `attention/gate_*` in
[`attention/__init__.py`](attention/__init__.py) and pass it to `run_linear_variant`.
Anything outside the linear/delta families builds a `Program` directly from the IR node
types, exactly as [`attention/_chunked.py`](attention/_chunked.py) and
[`pto_fuser.forwards.deltanet`](../src/pto_fuser/forwards) do.
