# pto-fuser — design

The auto-fuser: a graph-level scheduler that takes a sequence of tensor operations
(einsum contractions, opaque hand-optimized kernels, and Vector glue) and emits an
efficient execution plan over the [`pto-einsum`](../pto-einsum) substrate.

This document replaces the stale `PLANNER_DESIGN.md` that used to live in
`pto-einsum`. It states the thesis, the IR, the planner levers, the codegen
targets, the correctness discipline, and a staged milestone plan, with the full
chunked DeltaNet forward (`prototypes/deltanet_chunk/delta_e2e.py`) as the worked
example that the whole design is calibrated against.

This is the **target**; [`IMPLEMENTATION.md`](IMPLEMENTATION.md) records the
**realized state** and is kept in sync with it (currently: **M1 complete; M2 levers
2 & 3 complete**, lever 4 detected with codegen staged for M4; **M3 graph capture
complete** — staged chain captured + replayed as one dispatch, bit-exact, with the
launch-bound dispatch-elim win measured). Any divergence
between the two — notably that levers 2/3 are realized *inside* the soft-frozen
substrate and the fuser *selects among* them — is called out in that document's
sync ledger.

---

## 1. Thesis

Three claims, each validated by prototype or benchmark, define the bet:

1. **Fusion is a separate layer on top of `einsum()`, not baked into it.** The
   substrate compiles *one* contraction at a time and is soft-frozen. Everything
   graph-level — which read mode each stage uses, which stores are fused, which
   stages collapse into a single kernel, when to capture a dispatch graph — lives
   here.

2. **The whole chunk-attention taxonomy decomposes into substrate primitives plus
   a small set of opaque nodes.** Validated end-to-end:
   - **T0** (`kkt`): matmul-core + on-chip gated epilogue (`prototypes/kkt_fused/`).
   - **T2** (`chunk_h`): sequential cross-chunk recurrence with resident state that
     is *both* a matmul operand and a Vec accumulator (`prototypes/chunk_h_scan/`).
   - **T3** (DeltaNet/WY): an opaque triangular-inverse node bracketed by einsum
     stages — the only node that is *not* the matmul-core shape
     (`prototypes/deltanet_chunk/`). The full 6-stage forward is bit-faithful to an
     fp32 reference at the production shape (B8 H32 nc64 C64, M=16384, frob_rel
     ≤ 7e-4). **No missing capability — this is integration, not new primitives.**

3. **Fusion buys dispatch-elimination, not compute.** Measured repeatedly: at the
   production GDN shape, torch *glue* (cumsum, gating, masking, scaling) is 66–87%
   of each stage's cost and is HBM-bandwidth-bound; the contraction is the small
   part. Staged einsum stages + graph-captured dispatch **meet or beat** monolithic
   "mega" fusion, and mega fusion showed non-determinism that staged did not. So the
   planner's first-class lever is *dispatch-elim via graph capture*, and monolithic
   single-kernel fusion is reserved for the narrow launch-bound small-`T` regime
   where it provably wins (~2.4× at T=128, perf-neutral by T≥512).

The consequence: the fuser is **not** a megakernel generator. It is a scheduler
that (a) picks the cheapest substrate read/store mode per stage, (b) fuses
adjacent Vec glue into a contraction's epilogue/prologue when bandwidth-bound,
(c) eliminates per-stage launch cost with graph capture, and (d) falls back to
genuine single-kernel fusion only where the launch-bound regime justifies it.

---

## 2. Scope

**In scope:** the IR, the planner, codegen of fused nodes (epilogue/prologue Vec
ops onto a contraction; opaque-node hosting through GM), graph-capture emission, a
registry of opaque kernels, the staged-vs-fused correctness gate, and a small
library of reference chunk-attention forwards built on the above.

**Out of scope (lives in the substrate, soft-frozen):** the contraction codegen
itself — tile-matmul core, read modes (NT/NN/TN, §2.11–2.13), fused-output store
(§2.9), Vec elementwise/broadcast kernels, split-K, the transpose phases,
persistent-workspace dispatch. The fuser *selects among* these; it does not modify
them. If the fuser needs a substrate capability that does not exist, that is a
demand-driven substrate change (see `pto-einsum` roadmap), made deliberately and
behind a default-off flag, **not** a fuser-side workaround.

---

## 3. Intermediate representation

A program is a DAG of typed nodes over named tensors. Three node types — the
taxonomy proved these three suffice.

### 3.1 `EinsumNode` — a contraction stage
The core unit. Wraps one substrate `einsum()` call plus the substrate annotations
the planner is free to choose:

```
EinsumNode(
  equation:    str,                 # e.g. "nid,njd->nij"
  inputs:      [TensorRef, ...],
  output:      TensorRef,
  read_mode:   {NN, NT, NN_strided, TN},   # §2.11–2.13 — chosen by planner
  fuse_out:    bool,                # §2.9 fused permuted store, when free1-innermost
  epilogue:    [VecOp, ...] | None, # gating/mask/scale folded into the store
  prologue:    [VecOp, ...] | None, # per-operand scaling folded into the load
)
```

`read_mode`/`fuse_out`/`epilogue`/`prologue` are **planner outputs**, not user
input — the user writes the math, the planner annotates. The default (NN, no
fuse, no folded glue) is always a correct lowering; every annotation is an
optimization the planner must prove (gate) before keeping.

### 3.2 `OpaqueNode` — a foreign hand-optimized kernel
A node the matmul-core cannot express (today: the rec_unroll triangular inverse).
Hosted, not generated. Carries an explicit, pinned contract:

```
OpaqueNode(
  kernel:      registry key,        # e.g. "tri_inv_rec_unroll"
  inputs:      [TensorRef, ...],    # with required dtype + layout per slot
  output:      TensorRef,           # dtype + layout
  gm_contract: ...,                 # GM I/O shapes; AICORE device-fn vs __global__
  pre/post:    [VecOp, ...],        # adapters (e.g. transpose-to-upper for tri_inv)
)
```

The DeltaNet probe pinned exactly this: the tri-inv kernel is fp16-in/fp32-out,
inverts strictly-*upper* unit-triangular (so the adapter transposes the
strictly-lower `A`, and transposes the result back), `matrix_size ∈ {16,32,64,128}`,
and its core `run_tri_inv_rec_unroll<...>` is an **AICORE device fn** — sequenceable
as its own `.so` *and* inlinable into a fused driver later. The registry records
all of this so the planner can both stage it and (eventually) inline it.

**Dtype is part of the contract.** The single hardest e2e bug was handing the
fp16-only opaque kernel an fp32 tensor → it reinterpreted bytes → deterministic
NaN. The IR makes per-slot dtype explicit and the lowering inserts the cast; there
is no implicit dtype agreement. (See `pto-einsum/OPEN_QUESTIONS.md` for the
torch_npu format note that this finding *retired* — it was a dtype bug, not a
layout hazard.)

### 3.3 `VecGlueNode` — a standalone Vector op
Elementwise / broadcast / scan glue that is *not* (yet) folded into an adjacent
contraction: cumsum, gating `exp(g)`, causal mask, scale, `diag(β)·` scaling.
Lowers to the substrate's Vec kernels (elementwise Hadamard, broadcast/scaling) or,
for prefix scans, to a dedicated scan kernel. The planner's fusion pass tries to
*absorb* these into an adjacent `EinsumNode.epilogue`/`prologue`; a `VecGlueNode`
that survives the pass is launched on its own.

> **Note on cumsum.** The prefix-scan glue carries a known determinism edge
> (Vec→Scalar dependency; see OPEN_QUESTIONS). When the planner folds a scan into
> an epilogue or emits it standalone, the scan codegen must carry the
> `set_flag(PIPE_V, PIPE_S)` edge. This is a correctness invariant of the lowering,
> gated by a determinism test, not an optimization.

---

## 4. Planner levers

Ordered by the evidence for their payoff. Each lever is *opt-in per node* and must
pass the correctness gate (§6) against the default lowering before it is kept.

1. **Graph capture (dispatch-elim) — the #1 lever.** NPUGraph captures the PTO
   ctypes launches bit-exactly; replaying a captured graph collapses N per-stage
   dispatches into one. This is where staged *beats* mega at small `T` (2.4× at
   T=128) and is free/perf-neutral when compute-bound (T≥512). Applies to any
   chain of `EinsumNode`s + foldable glue whose shapes are static across replay.
   *Constraint:* the captured region must contain no per-call host sync — which is
   exactly why mega (per-call `cu.cpu()`) is **not** capturable and staged is.

2. **Read-mode selection (§2.11–2.13) — eliminates Phase A.** Pick NT / NN-strided
   / TN so both operands read straight from the raw tensors when the contraction
   axis is innermost/strided appropriately. 1.5–2.2× vs torch, 3.2–4.9× vs the
   Phase-A path, on the GDN shapes. Pure substrate selection; the planner reads the
   operand layouts and picks the mode (default NN is always valid).

3. **Fused-output store (§2.9) + operand-swap.** When the output's `free1` is
   innermost, the Cube stores the permuted tile straight to the result and Phase C
   is dropped (3.6–3.9× on `->bsht`). When it isn't, swapping the two operands can
   *make* it innermost (2.11× when it fires). The planner tries the swap, gates it,
   keeps it if it wins.

4. **Glue absorption (epilogue/prologue fusion).** Fold bandwidth-bound Vec glue
   into the adjacent contraction's store/load so the intermediate never round-trips
   HBM. This is the kkt_fused result: glue 32ms → 0.8ms (~40×) by killing the qk
   HBM round-trip. The reusable codegen unit is *tile-matmul core with a pluggable
   load front-end and a pluggable epilogue* — the fuser's main generated artifact.

5. **Resident-state scheduling (T2).** For sequential recurrences, schedule
   per-(b,h) so the carried state stays resident across chunks (operand + Vec
   accumulator) instead of round-tripping. Contained reorg at the fusion layer; the
   substrate primitives (in_nt=2 + in_nt=3) are unchanged.

6. **Selective monolithic fusion — last resort, narrow regime only.** Collapse a
   sub-chain into a single kernel (inlining an opaque AICORE device-fn through GM,
   cf. chunk_h_scan) *only* where launch-bound small-`T` justifies it and graph
   capture is insufficient. Always gated: mega showed non-determinism (root-caused
   to a missing cumsum Vec→Scalar edge) that staged did not — so a mega lowering is
   never kept without a determinism + frob_rel gate vs the staged lowering.

**Lever ordering rule:** prefer (1)+(2)+(3)+(4) — staged stages with the right
read/store modes, glue absorbed, dispatch captured — and only reach for (6) when a
measured launch-bound regime demands it. The default planner target is
"staged + captured," not "one big kernel."

---

## 5. Codegen / execution backends

- **Staged launch (baseline).** Each node → its own substrate `.so` (persistent
  workspace; setup/exec/teardown). Stages share GM tensors. Always available;
  the correctness reference for every other backend.
- **Graph-replay.** The staged chain wrapped in an NPUGraph capture; replay is one
  dispatch. The default production backend for static-shape chains.
- **Fused-node.** A generated kernel = matmul-core + pluggable load front-end +
  pluggable Vec epilogue/prologue (lever 4), optionally inlining an opaque AICORE
  device-fn (lever 6). This is the only backend that emits *new* device code; it
  reuses the substrate's tile-matmul core as a header-level building block, it does
  not fork it.

The substrate stays the single source of the matmul/Vec/transpose codegen. The
fuser composes and captures; the one place it generates is the fused-node epilogue/
prologue wiring around the borrowed core.

---

## 6. Correctness gating (non-negotiable)

Every non-default lowering is kept only if it passes, against the default staged
lowering on the same inputs:

- **`frob_rel` gate** — relative Frobenius norm of the difference under a fixed
  tolerance (the e2e example uses 2e-2 per stage; tighter for pure-matmul stages).
  Staged-vs-fused is *always* gated — measured "wins" of 0.93×/0.77× turned out to
  be broken (ungated zero/garbage pipelines) once gated.
- **Determinism gate** — run the candidate twice on identical input; bit-identical
  required. This is what caught the mega H=64 non-determinism (cumsum Vec→Scalar
  edge) that a single-run frob check masks. Mandatory on any mega/fused lowering and
  on any lowering that includes a scan.

The gate is part of the planner loop, not a separate test phase: a lever that does
not pass is discarded and the default lowering stands.

---

## 7. Worked example — chunked DeltaNet forward

`prototypes/deltanet_chunk/delta_e2e.py` is the reference the design is calibrated
to. The 6-stage forward, as IR:

| # | stage            | node                                    | levers in play |
|---|------------------|-----------------------------------------|----------------|
| 1 | `cumsum(g)`      | `VecGlueNode` (scan)                     | scan edge invariant; absorb→2 |
| 2 | `kkt → A`        | `EinsumNode` "nid,njd->nij" + tril epi   | NT read, gated epilogue (T0) |
| 3 | `solve_tril → T` | `OpaqueNode` tri_inv (fp16→fp32, ±adapt) | opaque host; dtype cast |
| 4 | `recompute W,U`  | `EinsumNode` "nij,njd->nid" ×2           | NN-strided; RHS-stack |
| 5 | `chunk_delta_h`  | `EinsumNode` scan (resident S)           | TN + NN-strided; resident-state (T2) |
| 6 | `chunk_o → o`    | `EinsumNode` + causal mask + scale       | NT read; gated epilogue (T0) |

Every row is a proven primitive class. The planner's job on this graph: pick read
modes (col. "levers"), absorb stages 1/2/6 glue into adjacent contractions, host
the opaque node in stage 3 with the dtype cast, schedule stage 5 per-(b,h)
resident, and graph-capture the whole static chain. The default staged lowering is
already bit-faithful (validated); each lever is then gated in.

---

## 8. Milestones

- **M1 — IR + staged executor + gate.** The three node types, a staged backend
  that reproduces `delta_e2e.py` from an IR description, and the frob_rel +
  determinism gate harness. Exit: DeltaNet forward runs from IR, staged, gated
  green at M=16384.
- **M2 — read-mode + fused-output + glue-absorption planner.** Levers 2/3/4 as
  gated annotations. The opaque-node registry (tri_inv first). Exit: each lever
  measured on the GDN/DeltaNet stages, kept only where gated-green and faster than
  default.
- **M3 — graph capture.** Lever 1: capture the staged chain, replay as one
  dispatch. Exit: DeltaNet/GDN forward at small `T` shows the dispatch-elim win
  (target ~2.4× at T=128), perf-neutral and still bit-exact at T≥512.
- **M4 — selective fused-node + resident-state.** Lever 6 (opaque AICORE inline +
  matmul-core epilogue codegen) and lever 5, *only* for the regimes M3 leaves on
  the table, each behind a measured launch-bound justification and a determinism
  gate. Exit: a documented decision per stage — staged-captured vs fused — with the
  measurement that chose it.

A second reference forward (GDN or KDA, which share the kkt/wy/chunk_h/chunk_o
stages) is added once M2 lands, to confirm the planner generalizes beyond the one
worked example before M3.

---

## 9. Risks / open items

- **Static-shape assumption for graph capture.** Replay needs shapes fixed across
  calls. Dynamic seqlen / chunk count breaks a single captured graph — likely a
  small family of captures keyed by shape bucket. Quantify before committing M3.
- **Opaque-node inlining ABI.** Lever 6 inlines an AICORE device-fn through GM; the
  chunk_h_scan prototype is the pattern, but reconciling the opaque kernel's build
  flags (`--cce-soc-version=Ascend910B4 --cce-soc-core-type=CubeCore`) with the
  substrate's (`--npu-arch=dav-2201`, same a2/a3 family) inside *one* `.so` is
  unproven — staged multi-`.so` sharing GM works today, single-`.so` inline does
  not yet. Treat as M4 research, not assumed.
- **Substrate gaps surfaced by the planner** are logged to the `pto-einsum`
  roadmap as demand-driven items (e.g. multi-axis free dims, strided-col fused
  store), not patched around in the fuser.
- **torch_npu / PTO-ISA oddities** go to `pto-einsum/OPEN_QUESTIONS.md` for external
  guidance; the fuser does not encode folklore around them.
