# pto-fuser — implementation

What is actually built, tracked against [`FUSER_DESIGN.md`](FUSER_DESIGN.md). The
design states the target; this document records the realized state per milestone.
**The two are kept in sync** — every change that lands a design element updates the
matching section here, and any divergence (a design idea not yet built, a built
thing not yet in the design) is called out explicitly below.

Status: **M1 complete. M2 levers 2 & 3 complete** (read-mode + operand-swap planner,
gated + measured on the DeltaNet and GDN stages); M2 lever 4 (glue absorption) is
**detected** here and its on-chip codegen is staged for M4. M3–M4 not started.

---

## Package layout

```
pto-fuser/
├── docs/
│   ├── FUSER_DESIGN.md     # the design (thesis, IR, levers, gating, milestones)
│   └── IMPLEMENTATION.md   # this file
├── src/pto_fuser/
│   ├── ir.py               # the IR: TensorRef + 3 compute nodes + TensorOp + Program
│   ├── registry.py         # opaque-kernel registry (tri_inv_rec_unroll)
│   ├── executor.py         # StagedExecutor + substrate_modes (honors the annotations)
│   ├── planner.py          # M2 Planner — gate-and-measure lever selection
│   ├── gate.py             # frob_rel + determinism gates
│   └── forwards/
│       ├── deltanet.py     # the DeltaNet forward as an IR Program (+ fp32 reference)
│       └── gdn.py          # the GDN contraction stages (2nd forward for the planner)
├── prototypes/             # the design proofs (T0/T2/T3 seed kernels)
├── run_deltanet.py         # M1 driver: build → run staged → gate (any shape)
├── run_plan.py             # M2 driver: plan → decision ledger → gate the annotated prog
└── tests/
    ├── test_ir.py          # IR structural validation (no NPU)
    ├── test_planner.py     # M2 planner keep-logic, injected measurements (no NPU)
    ├── test_deltanet_m1.py # M1 exit test: DeltaNet from IR, staged, gated (NPU)
    └── test_m2_npu.py      # M2: planner levers on DeltaNet + GDN stages, gated (NPU)
```

The package depends on the pinned `pto-einsum` substrate (sibling repo by default,
`PTO_EINSUM` override) and on `pto-kernels` for the opaque tri-inv (`PTO_KERNELS`).
No install is required — `src/` is added to `sys.path` by `tests/conftest.py` and
the driver, mirroring the prototype style. Runtime env as for the substrate
(`PTO_LIB_PATH`, `ASCEND_HOME_PATH`).

---

## M1 — IR + staged executor + gate  ✅

Exit criterion (design §8): *DeltaNet forward runs from IR, staged, gated green at
M=16384.* **Met** — frob_rel ≤ 7e-4 on all seven stages (tol 2e-2), bit-identical
across two runs, at B8 H32 nc64 C64 D128 (M=16384, 618 nodes).

### IR (`ir.py`) — design §3

The **three compute node types** are realized exactly as the design fixes them:

| node | realized as | notes |
|------|-------------|-------|
| `EinsumNode` | one substrate `einsum()` call | carries the planner-annotation fields (`read_mode`, `fuse_out`, `epilogue`, `prologue`) as design §3.1 specifies — **inert in M1** (the executor honors only the default NN/no-fuse lowering); M2 populates + gates them. `out_dtype` casts the fp32 substrate result. |
| `OpaqueNode` | a key into the registry | dtype is part of the contract — the registry lowering inserts the cast (design §3.2). |
| `VecGlueNode` | torch host op (`tril`/`add`/`sub`/`mul`/`scale`) | M1 lowers glue on the host; the substrate Vec kernels / glue-absorption (lever 4) are M2. Residuals accumulate in fp32, cast back via `out_dtype` (matches the reference numerics). |

A `Program` is an **ordered list** of steps (the unrolled static chain — exactly
the shape graph capture wants in M3) with declared `inputs`/`outputs`.
`Program.__post_init__` does a cheap use-before-def / undeclared-output check.

**One element beyond the design's three node types:** `TensorOp` — host tensor
plumbing (`reshape`/`contiguous`/`transpose`/`cast`/`slice`/`stack`/`zeros`). It is
**not** a fourth compute node: it carries no device kernel and exists only because
M1 is a host-driven staged executor that must thread intermediates between kernels.
The design's prose already assumes this plumbing (the prototype `delta_e2e.py` does
the same reshapes/slices/stacks inline in Python). Under graph capture (M3) these
collapse into buffer-binding metadata rather than ops. Flagged here as the one
implementation construct not named in the design; it does not change the IR's
compute surface.

### Opaque registry (`registry.py`) — design §3.2

`OpaqueRegistry` maps a kernel key → `(factory, lowering, contract)`. The factory
compiles the kernel lazily and caches it; the lowering owns the adapters + the
dtype cast; the `OpaqueContract` pins per-slot dtypes (documentation + the cast
rule). Seed entry **`tri_inv_rec_unroll`**: reuses the pto-kernels `fast_inverse`
JIT build, feeds the strictly-lower `A` transposed (kernel inverts strictly-upper
unit-triangular), casts to fp16 (kernel requirement — this is the dtype-is-the-
contract discipline that fixed the historical NaN), returns `(I+A)^-1` fp32. The
raw ctypes launch is bracketed by the two `npu.synchronize()` calls the contract
requires.

### Staged executor (`executor.py`) — design §5

`StagedExecutor.run(program, bindings)` walks the node list over a name→tensor
environment, dispatching each `EinsumNode` to its own substrate `.so` (persistent-
workspace setup/exec/teardown, owned by `pto_einsum`); stages share GM tensors
through the environment. This is the correctness reference every later backend
(graph-replay, fused-node) is gated against. The substrate `einsum` is resolved
lazily so the package imports off-NPU (the IR tests need no device).

### Gate harness (`gate.py`) — design §6

- `gate_frob_rel` / `gate_outputs` — relative Frobenius norm under a fixed
  tolerance, per named output.
- `gate_determinism` — runs a candidate twice, requires every output
  bit-identical (`torch.equal`). This is the guard that caught the historical mega
  H=64 non-determinism a single-run frob check masks; mandatory on any
  mega/fused/scan lowering as the levers land.

Gates return a `GateResult` (`passed` + human-readable detail), so they compose
into the planner loop later rather than being a separate test phase.

### Worked example (`forwards/deltanet.py`) — design §7

`build_deltanet_program(B,H,nc,C,D,scale)` emits the full 5-stage forward as a
`Program`: kkt einsum + tril glue → opaque tri-inv → recompute W,U → the
cross-chunk scan **unrolled over chunks** (each chunk: two einsums + sub/add glue,
state `S` threaded by name) → chunk_o (three einsums + tril/add/scale glue). Pure
DeltaNet (gate g = 0), matching the prototype. `deltanet_reference` is the fp32
torch pipeline (the gate reference); `make_inputs` reproduces the prototype's
random init.

The scan is unrolled in the builder (a host loop emitting per-chunk nodes), which
is the design's M1 "staged per-chunk einsum loop" (§7 row 5 / §8 M1). Resident-state
scheduling (lever 5) is M4.

### Verification

```bash
export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
pytest                                          # 5 passed (4 IR + 1 DeltaNet M1)
python run_deltanet.py --B 8 --H 32 --nc 64 --C 64   # M=16384 exit shape, ALL OK
```

Measured at the exit shape (M=16384): `A 2.9e-4 · T 5.6e-5 · W 3.0e-4 · U 3.0e-4 ·
h_state 6.8e-4 · v_new 4.5e-4 · o 7.0e-4`, determinism bit-identical.

---

## M2 — read-mode + fused-output planner  ✅ (levers 2 & 3);  lever 4 detected, codegen → M4

Exit criterion (design §8): *each lever measured on the GDN/DeltaNet stages, kept
only where gated-green and faster than default.* **Met for levers 2 & 3.**

### The key finding that shapes M2

Levers 2 (read-mode NT/NN-strided/TN, §2.11–2.13) and 3 (operand-swap → fused store,
§2.9) are **realized inside the soft-frozen substrate**: its recipe builder
auto-selects them from the equation + operand layout (`builder.py`, `in_nt` and
`_swap_operands`), and exposes the documented toggles `EINSUM_DISABLE_NT` /
`EINSUM_DISABLE_OPERAND_SWAP`. Per design §2 the fuser *selects among* substrate
capabilities — it does not re-implement them. So the M2 planner **measures which
lowering wins and records it**, rather than forcing a mode it re-derived. This is
on-plan (§2, §4: "Pure substrate selection; the planner reads the operand layouts
and picks the mode"), not a shortcut.

### Planner (`planner.py`) — design §4 levers 2/3, §6 gate, §8 M2

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
4. **keeps** a lever only when it *fired* (the substrate changed the lowering),
   *gated-green*, **and** *faster*; pins the kept choice onto `EinsumNode.read_mode`
   / `fuse_out`.

It returns the annotated `Program` plus a `LeverDecision` ledger (mode fired, frob,
both wall-clocks, keep/drop). The executor honors the annotations via
`substrate_modes` (a context manager over the two toggles); `read_mode="auto"` /
`fuse_out=True` set nothing — the substrate decides — so the default path is
unchanged from M1.

### Measured (B4 H16 nc8 C64 D128)

The lever's payoff is **layout-dependent**, and the planner's per-stage decision
reflects exactly that:

| forward | stage | equation | mode fired | speedup vs Phase-A | kept |
|---------|-------|----------|------------|--------------------|------|
| GDN | kkt     | `bihd,bjhd->bihj` | NT         | **5.3×**  | ✅ |
| GDN | wy_fast | `bihj,bjhd->bihd` | NN-strided | **10.3×** | ✅ |
| GDN | chunk_h | `bvhd,bvhe->bhde` | TN         | **2.7×**  | ✅ |
| GDN | chunk_o | `bvhd,bhde->bvhe` | NN-strided | **3.6×**  | ✅ |
| DeltaNet | kkt / W,U / scan / o | various | NT/NN/TN | ~1.0× | mixed |

The GDN family's head axis `h` is a non-innermost batch axis, so Phase-A pays a
large strided-gather cost the direct read eliminates — hence 2.7–10.3× (matching the
substrate's own §2.11–2.13 measurements). DeltaNet's contractions are flat
`[M,C,D]`, so Phase-A is already cheap and the direct read is ~1.0× (gated-green,
bit-identical, but not always faster) — the planner correctly keeps only the stages
that measure faster and drops the rest. **Every candidate gated frob_rel = 0.0**
(the direct reads are bit-identical to the baseline), and the lever-pinned DeltaNet
program still matches the fp32 reference (`gate_outputs` ALL OK). Operand-swap does
**not fire** on any DeltaNet/GDN stage — their outputs are already fusible (free1
innermost) — so it is correctly measured and dropped; it is retained for forwards
whose natural output needs a Phase-C transpose.

### Lever 4 (glue absorption) — detected here, codegen is M4

`Planner.absorption_candidates` finds `VecGlueNode → EinsumNode` adjacencies whose
intermediate round-trips HBM (e.g. the DeltaNet scan's `vn{c} → dS{c}`,
`S{c} → WS{c}`) — the foldable pairs lever 4 targets. The **actual on-chip fold**
(matmul-core + pluggable epilogue/prologue) is the *fused-node backend*, which design
§5 names as "the only backend that emits new device code"; it is staged for M4 on
the `kkt_fused` prototype (glue 32 ms → 0.8 ms, ~40×). M2 decides *that* a fold is
available; M4 emits it. This split is recorded in the sync ledger below.

### Second reference forward (`forwards/gdn.py`)

The four GDN contraction stages (kkt / wy_fast / chunk_h / chunk_o) as single-node
Programs with synthetic operands — a **different equation family** (head axis `h` is
a non-innermost batch axis) that exercises all three direct-read modes, confirming
the planner generalizes beyond DeltaNet (design §8: "a second reference forward …
to confirm the planner generalizes before M3"). The full GDN forward additionally
needs the gating cumsum, GQA repeat, and the chunk_h recurrence with resident state
(lever 5 = M4); those are glue around the same four contractions, so the contraction
stages are the right unit for the M2 lever decision.

### Verification

```bash
pytest                                                  # 13 passed (off-NPU + NPU)
python run_plan.py --B 4 --H 16 --nc 8 --C 64 --D 128   # decision ledger, ALL OK
```

## M3 — graph capture  ⬜ not started

Lever 1: wrap the staged chain in an NPUGraph capture, replay as one dispatch. The
`Program`'s ordered static node list is already the right representation; the
`TensorOp` plumbing becomes buffer-binding metadata. Static-shape / bucketing is
the open risk (design §9).

## M4 — selective fused-node + resident-state  ⬜ not started

Lever 6 (opaque AICORE inline + matmul-core epilogue codegen) and lever 5, only for
the regimes M3 leaves on the table, each behind a measured launch-bound justification
and a determinism gate.

---

## Sync ledger (design ↔ implementation)

Tracks where the realized state intentionally differs from the design, so the two
docs stay honest:

- **`TensorOp`** is in the implementation but not named among the design's three
  node types — by intent (host plumbing, not a compute type; see M1/IR above).
  Not a design change.
- **`EinsumNode` planner annotations** were inert in M1; **active in M2** — the
  Planner sets `read_mode` / `fuse_out` and the executor honors them. On-plan (§3.1).
- **Levers 2 & 3 are substrate-internal.** The read-mode (NT/NN-strided/TN) and
  operand-swap→fused-store selection lives in the soft-frozen substrate's recipe
  builder and auto-fires from the layout; the fuser *measures and selects* via the
  documented `EINSUM_DISABLE_NT` / `EINSUM_DISABLE_OPERAND_SWAP` toggles rather than
  re-deriving the mode. This is design §2/§4 as written ("Pure substrate selection;
  the planner reads the operand layouts and picks the mode") — **not** a divergence,
  but recorded because the design's prose can read as if the fuser computes the mode.
- **`read_mode` is two-valued** (`"auto"` | `"NN"`) in the implementation, where
  §3.1 lists `{NN, NT, NN_strided, TN}`. The substrate picks *which* direct-read mode
  fires; the fuser only chooses direct-read-vs-Phase-A, so the enum collapses to
  auto/NN. The fired mode is still recorded (in the `LeverDecision` ledger) for
  reporting. On-plan, narrower surface.
- **Lever 4 (glue absorption) is split M2/M4.** M2 *detects* foldable
  `VecGlueNode → EinsumNode` pairs (`absorption_candidates`); the on-chip *codegen*
  (matmul-core + epilogue) is the fused-node backend, which §5 names as the only
  backend emitting new device code and §8 places its build-flag reconciliation in M4
  (§9 risk). Glue still lowers host-side (M1 path) until then. On-plan.
- **Second forward is GDN *contraction stages*, not the full GDN forward.** The four
  contractions exercise every read-mode lever; the gating cumsum / GQA repeat /
  chunk_h resident-state recurrence (lever 5) are M4. On-plan (§8 says "GDN/DeltaNet
  stages").
- No design element is contradicted by the implementation.
