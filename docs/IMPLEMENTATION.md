# pto-fuser — implementation

What is actually built, tracked against [`FUSER_DESIGN.md`](FUSER_DESIGN.md). The
design states the target; this document records the realized state per milestone.
**The two are kept in sync** — every change that lands a design element updates the
matching section here, and any divergence (a design idea not yet built, a built
thing not yet in the design) is called out explicitly below.

Status: **M1 complete.** M2–M4 not started.

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
│   ├── executor.py         # StagedExecutor — the M1 baseline backend
│   ├── gate.py             # frob_rel + determinism gates
│   └── forwards/
│       └── deltanet.py     # the DeltaNet forward as an IR Program (+ fp32 reference)
├── prototypes/             # the design proofs (T0/T2/T3 seed kernels)
├── run_deltanet.py         # M1 driver: build → run staged → gate (any shape)
└── tests/
    ├── test_ir.py          # IR structural validation (no NPU)
    └── test_deltanet_m1.py # M1 exit test: DeltaNet from IR, staged, gated (NPU)
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

## M2 — read-mode + fused-output + glue-absorption planner  ⬜ not started

Levers 2/3/4 as gated annotations on `EinsumNode`; the opaque-node registry already
exists (M1) and grows as needed. The annotation fields are already in the IR (inert)
so M2 is: a planner pass that sets them, an executor that honors them, and the gate
in the loop. A second reference forward (GDN/KDA) is added here to confirm the
planner generalizes before M3.

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
- **`EinsumNode` planner annotations** exist in the IR but are inert until M2 — the
  design (§3.1) explicitly specifies them as planner outputs with the default
  lowering always valid, so this is on-plan, not a gap.
- **Glue lowering** is host-side torch in M1; the design's substrate-Vec / absorption
  path is M2 (lever 4). On-plan.
- No design element is contradicted by the implementation. When M2 starts, this
  ledger gets the read-mode/fuse/absorption entries.
