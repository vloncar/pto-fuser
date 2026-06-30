# pto-fuser — implementation

What is actually built, tracked against [`FUSER_DESIGN.md`](FUSER_DESIGN.md). The
design states the target; this document records the realized state, feature by
feature. **The two are kept in sync** — every change that lands a design element
updates the matching section here, and any divergence (a design idea not yet built, a
built thing not yet in the design) is called out explicitly in the sync ledger below.

Status — all features are built and gated:

- **IR + staged executor + gate** — the three node types, the staged backend, and the
  frob_rel + determinism gate harness.
- **Read-mode / fused-output planner** — read-mode selection and operand-swap →
  fused-store, gated + measured on the DeltaNet and GDN stages; glue absorption is
  **detected** by the planner and its on-chip fold is **realized by the fused-node
  backend**.
- **Graph capture** — the staged chain captured into one NPUGraph and replayed as a
  single dispatch, bit-exact, with the launch-bound dispatch-elimination win measured.
- **Fused-node backend + fusion decision** — the resident-state scan and the gated-kkt
  kernels hosted as single-dispatch nodes, with the staged-vs-fused decision keeping a
  fused lowering only where it gates bit-faithful + deterministic AND beats
  staged-captured (measured per stage).

---

## Package layout

```
pto-fuser/
├── docs/
│   ├── FUSER_DESIGN.md     # the design (thesis, IR, optimization features, gating)
│   └── IMPLEMENTATION.md   # this file
├── src/pto_fuser/
│   ├── ir.py               # the IR: TensorRef + 3 compute nodes + TensorOp + Program
│   ├── registry.py         # opaque-kernel registry (tri_inv_rec_unroll)
│   ├── executor.py         # StagedExecutor + library_modes (honors the annotations)
│   ├── planner.py          # Planner — gate-and-measure read-mode / fused-store selection
│   ├── graph.py            # graph-replay backend (CaptureExecutor + GraphReplayExecutor)
│   ├── fused.py            # fused-kernel registry (FusedKernel persistent runners)
│   ├── fusion.py           # the staged-vs-fused decision (FusionDecision: gate + measure)
│   ├── gate.py             # frob_rel + determinism gates
│   ├── kernels/            # the hosted fused device kernels (chunk_h_scan, kkt_fused)
│   └── forwards/
│       ├── deltanet.py     # the DeltaNet forward as an IR Program (+ fp32 reference)
│       ├── gdn.py          # the GDN contraction stages (2nd forward for the planner)
│       └── fused_stages.py # staged Program + FusedNode + reference, per fused stage
├── examples/               # runnable demos (see examples/README.md)
│   ├── common/             # shared device / timing / plotting infra
│   ├── minimal.py          # the smallest program
│   ├── attention/          # the chunked-attention zoo (linear family + GDN/KDA)
│   ├── workflow/           # one demo per feature (read_modes, graph_capture, fusion_decision)
│   └── benchmarks/         # per-feature GDN benchmark (table + plot)
└── tests/
    ├── test_ir.py          # IR structural validation (no NPU)
    ├── test_planner.py     # planner keep-logic, injected measurements (no NPU)
    ├── test_graph.py       # capture-mode toggle + replay-guard (no NPU)
    ├── test_fusion.py      # FusedNode IR + registry + decision logic (no NPU)
    ├── test_forwards.py    # chunked-attention decomposition vs recurrent ref (CPU)
    ├── test_examples.py    # every example imports + builds its Program (no NPU)
    ├── test_forwards_npu.py# DeltaNet from IR, staged, gated (NPU)
    ├── test_planner_npu.py # read-mode / fused-store selection on DeltaNet + GDN (NPU)
    ├── test_graph_npu.py   # capture/replay bit-exact + determinism + dispatch win (NPU)
    └── test_fusion_npu.py  # fused scan/kkt correctness + determinism + decision (NPU)
```

The package depends on the pinned `pto-einsum` library (sibling repo by default,
`PTO_EINSUM` override) and on `pto-kernels` for the opaque tri-inv (`PTO_KERNELS`).
No install is required — `src/` (and `examples/`) are added to `sys.path` by
`tests/conftest.py` and by the examples' bootstrap. Runtime env as for the library
(`PTO_LIB_PATH`, `ASCEND_HOME_PATH`). The NPU tests (`*_npu.py`, also `npu`-marked)
are **not collected unless `PTO_RUN_NPU=1`** — so the default `pytest` is the
device-free suite (importing `torch_npu` would touch the device); set `PTO_RUN_NPU=1`
on a healthy box to add the on-device forward/feature tests.

---

## IR + staged executor + gate

Validated: the DeltaNet forward runs from IR, staged, gated green at M=16384 —
frob_rel ≤ 7e-4 on all seven stages (tol 2e-2), bit-identical across two runs, at
B8 H32 nc64 C64 D128 (M=16384, 618 nodes).

### IR (`ir.py`) — design §3

The **three compute node types** are realized exactly as the design fixes them:

| node | realized as | notes |
|------|-------------|-------|
| `EinsumNode` | one library `einsum()` call | carries the planner-annotation fields (`read_mode`, `fuse_out`, `epilogue`, `prologue`) as design §3.1 specifies — **inert until the planner runs** (the executor honors only the default NN/no-fuse lowering); the planner populates + gates them. `out_dtype` casts the fp32 library result. |
| `OpaqueNode` | a key into the registry | dtype is part of the contract — the registry lowering inserts the cast (design §3.2). |
| `VecGlueNode` | torch host op (`tril`/`add`/`sub`/`mul`/`scale`) | the staged executor lowers glue on the host; the library Vec kernels / glue absorption are the fused-node backend's job. Residuals accumulate in fp32, cast back via `out_dtype` (matches the reference numerics). |

A `Program` is an **ordered list** of steps (the unrolled static chain — exactly the
shape graph capture wants) with declared `inputs`/`outputs`.
`Program.__post_init__` does a cheap use-before-def / undeclared-output check.

**One element beyond the design's three node types:** `TensorOp` — host tensor
plumbing (`reshape`/`contiguous`/`transpose`/`permute`/`cast`/`slice`/`stack`/`zeros`).
It is **not** a fourth compute node: it carries no device kernel and exists only
because the staged executor is host-driven and must thread intermediates between
kernels. Under graph capture these collapse into buffer-binding metadata rather than
ops. Flagged here as the one implementation construct not named in the design; it does
not change the IR's compute surface.

### Opaque registry (`registry.py`) — design §3.2

`OpaqueRegistry` maps a kernel key → `(factory, lowering, contract)`. The factory
compiles the kernel lazily and caches it; the lowering owns the adapters + the dtype
cast; the `OpaqueContract` pins per-slot dtypes (documentation + the cast rule). Seed
entry **`tri_inv_rec_unroll`**: reuses the pto-kernels `fast_inverse` JIT build, feeds
the strictly-lower `A` transposed (kernel inverts strictly-upper unit-triangular),
casts to fp16 (kernel requirement — the dtype-is-the-contract discipline that fixed
the historical NaN), returns `(I+A)^-1` fp32. The raw ctypes launch is bracketed by
the two `npu.synchronize()` calls the contract requires.

### Staged executor (`executor.py`) — design §5

`StagedExecutor.run(program, bindings)` walks the node list over a name→tensor
environment, dispatching each `EinsumNode` to its own library `.so` (persistent-
workspace setup/exec/teardown, owned by `pto_einsum`); stages share GM tensors through
the environment. This is the correctness reference every later backend (graph-replay,
fused-node) is gated against. The library `einsum` is resolved lazily so the package
imports off-NPU (the IR tests need no device).

### Gate harness (`gate.py`) — design §6

- `gate_frob_rel` / `gate_outputs` — relative Frobenius norm under a fixed tolerance,
  per named output.
- `gate_determinism` — runs a candidate twice, requires every output bit-identical
  (`torch.equal`). This is the guard that caught the historical mega H=64
  non-determinism a single-run frob check masks; mandatory on any mega/fused/scan
  lowering.

Gates return a `GateResult` (`passed` + human-readable detail), so they compose into
the planner loop rather than being a separate test phase.

### Worked example (`forwards/deltanet.py`) — design §7

`build_deltanet_program(B,H,nc,C,D,scale)` emits the full 5-stage forward as a
`Program`: kkt einsum + tril glue → opaque tri-inv → recompute W,U → the cross-chunk
scan **unrolled over chunks** (each chunk: two einsums + sub/add glue, state `S`
threaded by name) → chunk_o (three einsums + tril/add/scale glue). Pure DeltaNet
(gate g = 0). `deltanet_reference` is the fp32 torch pipeline (the gate reference);
`make_inputs` reproduces the random init.

The scan is unrolled in the builder (a host loop emitting per-chunk nodes), the
default "staged per-chunk einsum loop" (§7 row 5). Resident-state scheduling is the
fused-node backend's job.

### Verification

```bash
export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
pytest                                  # off-NPU suite (IR/planner/graph/fusion/forwards/examples)
python examples/attention/gdn.py                     # DeltaNet/GDN forward, staged, gated
```

Measured at the exit shape (M=16384): `A 2.9e-4 · T 5.6e-5 · W 3.0e-4 · U 3.0e-4 ·
h_state 6.8e-4 · v_new 4.5e-4 · o 7.0e-4`, determinism bit-identical.

---

## Read-mode / fused-output planner

Validated: each feature measured on the GDN/DeltaNet stages, kept only where
gated-green and faster than the default.

### The key finding that shapes the planner

Read-mode selection (NT/NN-strided/TN, §2.11–2.13) and operand-swap → fused store
(§2.9) are **realized inside the soft-frozen library**: its recipe builder
auto-selects them from the equation + operand layout (`builder.py`, `in_nt` and
`_swap_operands`), and exposes the documented toggles `EINSUM_DISABLE_NT` /
`EINSUM_DISABLE_OPERAND_SWAP`. Per design §2 the fuser *selects among* library
capabilities — it does not re-implement them. So the planner **measures which lowering
wins and records it**, rather than forcing a mode it re-derived. This is on-plan (§2,
§4: "Pure library selection; the planner reads the operand layouts and picks the
mode"), not a shortcut.

### Planner (`planner.py`) — design §4 (read-mode, fused-store), §6 gate

`Planner.plan(program, bindings)`:
1. runs the program once (staged, auto modes) to materialize every node's **real**
   operands;
2. for each *distinct* contraction (deduped by equation+shapes — the DeltaNet scan
   repeats one shape `nc` times, so a 600-node program costs a handful of builds),
   builds + times three lowerings on those operands: the always-valid **Phase-A NN
   baseline**, the **direct-read** candidate (NT/NN-strided/TN), and the
   **operand-swap** candidate;
3. **frob-gates** each candidate ≡ baseline (a broken lowering that produced
   zero/garbage fails here — design §6) and times it;
4. **keeps** a feature only when it *fired* (the library changed the lowering),
   *gated-green*, **and** *faster*; pins the kept choice onto `EinsumNode.read_mode`
   / `fuse_out`.

It returns the annotated `Program` plus a `LeverDecision` ledger (mode fired, frob,
both wall-clocks, keep/drop). The executor honors the annotations via `library_modes`
(a context manager over the two toggles); `read_mode="auto"` / `fuse_out=True` set
nothing — the library decides — so the default path is unchanged.

### Measured (B4 H16 nc8 C64 D128)

The payoff is **layout-dependent**, and the planner's per-stage decision reflects
exactly that:

| forward | stage | equation | mode fired | speedup vs Phase-A | kept |
|---------|-------|----------|------------|--------------------|------|
| GDN | kkt     | `bihd,bjhd->bihj` | NT         | **5.3×**  | ✅ |
| GDN | wy_fast | `bihj,bjhd->bihd` | NN-strided | **10.3×** | ✅ |
| GDN | chunk_h | `bvhd,bvhe->bhde` | TN         | **2.7×**  | ✅ |
| GDN | chunk_o | `bvhd,bhde->bvhe` | NN-strided | **3.6×**  | ✅ |
| DeltaNet | kkt / W,U / scan / o | various | NT/NN/TN | ~1.0× | mixed |

The GDN family's head axis `h` is a non-innermost batch axis, so Phase-A pays a large
strided-gather cost the direct read eliminates — hence 2.7–10.3× (matching the
library's own §2.11–2.13 measurements). DeltaNet's contractions are flat `[M,C,D]`, so
Phase-A is already cheap and the direct read is ~1.0× (gated-green, bit-identical, but
not always faster) — the planner correctly keeps only the stages that measure faster
and drops the rest. **Every candidate gated frob_rel = 0.0** (the direct reads are
bit-identical to the baseline), and the annotated DeltaNet program still matches the
fp32 reference (`gate_outputs` ALL OK). Operand-swap does **not fire** on any
DeltaNet/GDN stage — their outputs are already fusible (free1 innermost) — so it is
correctly measured and dropped; it is retained for forwards whose natural output needs
a Phase-C transpose.

### Glue absorption — detected here, folded by the fused-node backend

`Planner.absorption_candidates` finds `VecGlueNode → EinsumNode` adjacencies whose
intermediate round-trips HBM (e.g. the DeltaNet scan's `vn{c} → dS{c}`,
`S{c} → WS{c}`) — the foldable pairs glue absorption targets. The **actual on-chip
fold** (matmul-core + pluggable epilogue/prologue) is the *fused-node backend*, which
design §5 names as "the only backend that emits new device code"; it is realized on
the `kkt_fused` kernel (glue 32 ms → 0.8 ms, ~40×). The planner decides *that* a fold
is available; the fused-node backend emits it. This split is recorded in the sync
ledger below.

### Second reference forward (`forwards/gdn.py`)

The four GDN contraction stages (kkt / wy_fast / chunk_h / chunk_o) as single-node
Programs with synthetic operands — a **different equation family** (head axis `h` is a
non-innermost batch axis) that exercises all three direct-read modes, confirming the
planner generalizes beyond DeltaNet. The full GDN forward additionally needs the
gating cumsum, GQA repeat, and the chunk_h recurrence with resident state; those are
glue around the same four contractions, so the contraction stages are the right unit
for the read-mode / fused-store decision.

### Verification

```bash
pytest tests/test_planner.py            # keep-logic, injected measurements
python examples/workflow/read_modes.py --B 4 --H 16 --nc 8   # decision ledger, ALL OK
```

---

## Graph capture (dispatch elimination)

Wrap the staged chain in an NPUGraph capture and replay it as a single dispatch. The
`Program`'s ordered static node list is already the right representation; `graph.py`
adds the backend.

**Two prerequisites, both supplied by the backend.** A region is capturable only if it
contains (a) no JIT/codegen and (b) no per-call host sync:

1. **No JIT inside capture.** The library's one-shot `einsum()` *rebuilds* the kernel
   every call (codegen + dlopen + first-call workspace setup — all host work).
   `CaptureExecutor` (a `StagedExecutor` subclass) instead builds a **persistent**
   `EinsumBuilder(...).build()` runner per node *once* — keyed by `(equation, shapes,
   dtype, read_mode, fuse_out)` so the planner's annotations are honored at build time
   — and reuses it. The captured region is then pure device launches. (This
   persistent-runner path is also the fair dispatch baseline: the win measured is
   *graph vs persistent-staged*, not vs the rebuild-every-call convenience.)
2. **No host sync.** The opaque tri_inv lowering syncs to order its raw launch against
   the torch stream. `registry.capture_mode()` (a module flag, set by the backend
   **inside the capture region only**) drops those two `synchronize()`s; the launch is
   recorded on the capture stream in order, so they are both unnecessary and
   capture-breaking. This is what makes the *full* DeltaNet forward — opaque node
   included — capture as one graph. (Mega is not capturable for the dual reason: a
   per-call `cu.cpu()` host read; precisely the overhead staged-captured removes.)

**`GraphReplayExecutor`** — `capture(program, bindings)` clones the inputs into static
buffers, warms up eagerly (builds every runner, runs the one-time workspace setup,
primes caches), then captures one `run(...)` into an `NPUGraph`, keeping references to
the pool-resident output tensors. `replay(bindings)` copies the new operands into the
static buffers, replays, and returns the outputs (cloned by default; `clone=False` in
tight measurement loops). A replay whose shape differs from capture raises —
static-shape is enforced, not silently wrong (design §9).

### Measured — DeltaNet forward, graph-replay vs persistent-staged

The "T" axis is the chunk count (`T = nc·C`); ms/call over back-to-back launches + one
trailing sync (what exposes host dispatch). Every row bit-exact vs staged
(`frob_rel = 0`) and gated vs the fp32 reference.

| regime | shape | T | staged | graph | speedup |
|--------|-------|---|-------:|------:|--------:|
| launch-bound | B2 H4 | 64 (nc1) | 0.85 ms | 0.30 ms | **2.9×** |
| launch-bound | B2 H4 | 128 (nc2) | 1.54 ms | 0.38 ms | **4.0×** |
| launch-bound | B2 H4 | 1024 (nc16) | 3.23 ms | 1.33 ms | **2.4×** |
| crossover | B8 H32 | 64 (nc1) | 1.02 ms | 0.89 ms | 1.15× |
| compute-bound | B8 H32 | 256 (nc4) | 3.49 ms | 3.48 ms | 1.00× |
| compute-bound | B8 H32 | 2048 (nc32) | 34.8 ms | 35.6 ms | 0.98× |

The dispatch-elimination win is **regime-specific**, exactly as design §4 (graph
capture) predicts: a real multiplier where host launch dominates (small batch, and/or
few large chunks), and free/perf-neutral (within timing jitter) once device work per
launch hides the dispatch. Small-batch DeltaNet stays launch-bound across the whole nc
sweep because each unrolled scan stage is a tiny matmul; larger `B·H` crosses to
compute-bound by `nc≈4`. Graph capture never meaningfully regresses — so it is the
right **default** backend for any static-shape chain. Each GDN contraction stage also
captures and replays bit-exact (the direct-read equation family), confirming the
backend generalizes beyond the worked example.

### Verification

```bash
pytest tests/test_graph.py              # capture-mode toggle + replay guard
python examples/workflow/graph_capture.py --B 2 --H 4 --nc 1 2 8   # dispatch-elim sweep
```

---

## Fused-node backend + fusion decision

Validated: a documented decision per stage — staged-captured vs fused — with the
measurement that chose it. The fused-node backend hosts two proven kernels as
single-dispatch nodes, and the decision procedure keeps each only where it gates
bit-faithful + deterministic AND beats staged-captured.

### What graph capture leaves on the table

Graph capture removes per-stage *dispatch* cost but **not** the HBM round-trip of
intermediates *between* stages. Two stages pay that round-trip heavily, and they are
the two fusion features:

- **resident state (`chunk_h_scan`)** — the staged scan unrolls the cross-chunk
  recurrence into `nc` per-chunk matmul pairs and writes the carried state `S` back to
  HBM every chunk; the fused kernel keeps `S` resident (a matmul operand *and* a Vec
  accumulator) across all chunks.
- **glue absorption (`kkt_gated`)** — the staged kkt lands the qk matrix in HBM, then
  a Vec epilogue reads it back to gate + mask; the fused kernel folds the gated/masked
  epilogue into the matmul store, so qk never leaves on-chip.

### Fused-node backend (`fused.py`, IR `FusedNode`, executor branch)

A **`FusedNode`** (the one IR node with an `outputs` *list* — a fused kernel may
produce several, e.g. the scan's per-chunk readouts + final state) names a kernel in
the **`FusedKernelRegistry`**. The registry lazily compiles the kernel `.cpp`
(`src/pto_fuser/kernels/`) into a cached `.so` keyed by its compile-time shape
(`KKT_NC`/`KKT_H`; `SCAN_B/H/NC`) and wraps it in a **persistent `FusedKernel`**:
`setup` allocates the on-chip-state workspace once, `run` is a pure stream launch (no
host sync), `teardown` frees it. The launch is sync-free precisely so a `FusedNode` is
**graph-capturable** — the `CaptureExecutor` runs it unchanged (single dispatch
already; capture removes even that one host launch). Operand layout/dtype adapters live
in the registered lowering, exactly as the opaque registry does for tri_inv.

The hosted kernels are the proven artifacts in `src/pto_fuser/kernels/`
(`kkt_fused`, `chunk_h_scan`), built as their **own `.so` sharing GM** with the
surrounding stages — the form design §9 records as *working today*. The further step of
inlining an opaque AICORE device-fn into **one** `.so` with the library matmul core
(the tri_inv case, §9 build-flag reconciliation) stays **unproven research and is not
used here** — see the sync ledger.

### Decision procedure (`fusion.py`) — design §4 (monolithic fusion), §6, §8

`decide(stage, kernel, staged, fused)` runs both lowerings on identical inputs and
returns a `FusionDecision`:
- **frob gate** — fused output ≡ staged output (design §6: a broken fused pipeline
  fails here);
- **determinism gate** — fused run twice, bit-identical (design §6: mandatory on any
  fused/scan lowering — the guard that caught the historical mega H=64 NDET);
- **measurement** — both backends timed back-to-back + one trailing sync.

The fused lowering is **kept only if** gated-green, deterministic, *and* faster;
otherwise the staged-captured lowering stands (the feature-ordering rule — reach for a
fused kernel only on a measured win).

### Measured (frob ≡ staged, determinism, and ms/call; healthy NPU)

| stage | feature | shape | staged-captured | fused | speedup | frob | det | kept |
|-------|---------|-------|----------------:|------:|--------:|-----:|:---:|:----:|
| `chunk_h_scan` | resident state | B1 H4 nc8   | 0.504 ms | 0.114 ms | **4.40×** | 0.0 | ✅ | ✅ |
| `chunk_h_scan` | resident state | B8 H32 nc16 | 8.489 ms | 3.722 ms | **2.28×** | 0.0 | ✅ | ✅ |
| `kkt_gated`    | glue absorption | nc8 H4 (vs torch-staged) | 0.219 ms | 0.112 ms | **1.96×** | 3.3e-6 | ✅ | ✅ |

The scan's fused output is **bit-identical** to the staged lowering (`frob = 0.0`, both
eager and captured) and both match the fp32 reference (h_out 2.7e-4, final 3.3e-4).
Notably the resident-state win is **not** confined to the launch-bound regime: it holds
at B8 H32 (2.28×) because what it removes is the per-chunk *HBM round-trip of `S`*, a
bandwidth cost graph capture cannot touch — so unlike monolithic fusion (design §4,
"narrow regime only"), resident state fuses broadly. The kkt fold likewise wins by
keeping qk on-chip. Both decisions: **FUSE**.

### Verification

```bash
pytest tests/test_fusion.py             # FusedNode IR + registry + decision logic
python examples/workflow/fusion_decision.py --B 1 --H 4 --nc 8       # both stages, gated decision
python examples/benchmarks/gdn_features.py --B 1 --H 4 --nc 8        # per-feature table + plot
```

The examples auto-select a healthy NPU (the box is shared — a chip pinned by a neighbor
job, or wedged by an aicore timeout, is skipped).

---

## Sync ledger (design ↔ implementation)

Tracks where the realized state intentionally differs from the design, so the two docs
stay honest:

- **`TensorOp`** is in the implementation but not named among the design's three node
  types — by intent (host plumbing, not a compute type; see the IR section above). Not
  a design change.
- **`EinsumNode` planner annotations** were initially inert; **active once the planner
  runs** — the Planner sets `read_mode` / `fuse_out` and the executor honors them.
  On-plan (§3.1).
- **Read-mode + fused-store selection is library-internal.** The read-mode
  (NT/NN-strided/TN) and operand-swap → fused-store selection lives in the soft-frozen
  library's recipe builder and auto-fires from the layout; the fuser *measures and
  selects* via the documented `EINSUM_DISABLE_NT` / `EINSUM_DISABLE_OPERAND_SWAP`
  toggles rather than re-deriving the mode. This is design §2/§4 as written ("Pure
  library selection; the planner reads the operand layouts and picks the mode") —
  **not** a divergence, but recorded because the design's prose can read as if the
  fuser computes the mode.
- **`read_mode` is two-valued** (`"auto"` | `"NN"`) in the implementation, where §3.1
  lists `{NN, NT, NN_strided, TN}`. The library picks *which* direct-read mode fires;
  the fuser only chooses direct-read-vs-Phase-A, so the enum collapses to auto/NN. The
  fired mode is still recorded (in the `LeverDecision` ledger) for reporting. On-plan,
  narrower surface.
- **Glue absorption is split: detected by the planner, folded by the fused-node
  backend.** The planner *detects* foldable `VecGlueNode → EinsumNode` pairs
  (`absorption_candidates`); the on-chip fold (matmul-core + epilogue) is the
  **fused-node backend** — realized as the `kkt_gated` `FusedNode` (qk + gated/masked
  epilogue in one kernel, 1.96× and qk never lands in HBM). The fold is *hosted* from a
  proven kernel, not yet auto-generated from an `EinsumNode.epilogue` op list (see next
  entry). On-plan.
- **Second forward is GDN *contraction stages*, not the full GDN forward.** The four
  contractions exercise every read-mode feature; the gating cumsum / GQA repeat /
  chunk_h resident-state recurrence are the fused-node backend's concern. On-plan (§8
  says "GDN/DeltaNet stages").
- **Graph capture needs a persistent-runner executor the design does not name.** Design
  §5 describes graph-replay as "the staged chain wrapped in an NPUGraph capture," but
  the staged executor's one-shot `einsum()` rebuilds per call (host work) and is not
  capturable. `CaptureExecutor` (persistent runners, planner annotations honored at
  build) is the realized form of "the staged chain" for capture — an implementation
  necessity, not a design change. It is also the fair dispatch baseline.
- **Opaque nodes are captured by dropping their stream syncs.** §4 (graph capture) says
  "the captured region must contain no per-call host sync"; the tri_inv lowering has
  two. `registry.capture_mode()` drops them inside the capture region only (the raw
  launch is recorded on the capture stream in order) — so the *full* DeltaNet, opaque
  node included, captures. On-plan: it realizes the §4 constraint rather than excluding
  the opaque node from capture.
- **Graph capture is the default backend, measured.** §5 calls graph-replay "the
  default production backend for static-shape chains"; it never regresses (perf-neutral
  compute-bound, 2.4–4.0× launch-bound) and enforces the §9 static-shape assumption with
  an explicit shape-mismatch error on replay. Shape *bucketing* (a family of captures)
  is still deferred — single-shape capture is what is built.
- **Fused nodes are *hosted kernels*, not yet IR-driven codegen.** Design §5 names the
  fused-node backend "the only backend that emits *new* device code." It is realized by
  hosting the two proven hand-written kernels (`kkt_fused`, `chunk_h_scan`) as
  `FusedNode`s with persistent runners — the device code exists (the kernels *are* the
  codegen reference), but it is selected by registry key, not templated from an
  `EinsumNode.epilogue`/`prologue` op list. The decision procedure, the gate discipline,
  the capturable single-dispatch backend, and the per-stage measurement are all
  realized; *automatic* epilogue templating is the remaining codegen step. Recorded so
  "emits new device code" is not over-read as "generates it from the IR."
- **Single-`.so` opaque inline (monolithic fusion's hardest form) is NOT attempted — by
  design.** §9 flags reconciling the opaque AICORE kernel's build flags with the
  library's inside one `.so` as research, not assumed. Only the form §9 says works today
  is used: each fused kernel is its own `.so` sharing GM with the staged stages around
  it (and the opaque tri_inv stays a separate hosted node). The inline ABI is left as
  open research — claimed nowhere. On-plan (honours the §9 risk boundary).
- **`FusedNode` is the one multi-output IR node.** §3 fixes three single-output compute
  node types; a fused kernel legitimately produces several tensors (the scan's
  per-chunk readouts + final state), so `FusedNode` carries an `outputs` list and the
  executor updates the env with all of them (`node_outputs` / dict result). A backend
  detail, not a new compute surface — the three node *types* are unchanged.
- **Resident state wins beyond the launch-bound regime.** Design §4 frames the fused
  path as "narrow regime only" (true for monolithic fusion). The scan measurement shows
  resident state fuses at B8 H32 too (2.28×), because it removes the per-chunk *HBM
  round-trip of `S`* — a bandwidth cost, not a dispatch cost. The decision procedure
  keeps it on the measurement, exactly as intended; recorded as a sharper-than-designed
  finding, not a contradiction.
- No design element is contradicted by the implementation.
