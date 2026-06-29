# pto-fuser

The **fusion / auto-fuser layer** on top of the [`pto-einsum`](../pto-einsum)
contraction substrate. `pto-einsum` compiles individual tensor contractions to
Ascend Cube/Vec kernels and is *soft-frozen*; `pto-fuser` consumes it as a pinned
dependency and is where the *graph-level* work lives: scheduling sequences of
einsum-core stages, opaque foreign nodes, and Vec glue into efficient kernels.

The guiding thesis (validated across the chunk-attention taxonomy): **fusion is a
separate layer on top of the einsum primitives, not baked into `einsum()`.** The
reusable unit is a tile-matmul core with pluggable load front-ends + epilogue, plus
a registry of opaque hand-optimized nodes (e.g. triangular inverse) that the graph
can host but that are not the matmul-core shape.

## Layout

- `docs/FUSER_DESIGN.md` — the auto-fuser design (thesis, IR, planner levers,
  codegen targets, correctness gating, milestones).
- `docs/IMPLEMENTATION.md` — what is actually built so far, tracked against the
  design. The two are kept in sync: the design states the target, the
  implementation doc records the realized state per milestone.
- `src/pto_fuser/` — the package: the IR, the staged executor, the opaque-kernel
  registry, the correctness gates, the read-mode/fused-store **Planner** (`planner.py`),
  and the reference forwards (DeltaNet, GDN contraction stages) built on them.
  Drivers: `run_deltanet.py` (M1, staged + gate) and `run_plan.py` (M2, plan +
  decision ledger).
- `prototypes/` — the design proofs that established the substrate spans every
  chunk-attention family. These are the seed reference kernels for the fuser:
  - `kkt_fused/` — **T0**: einsum-core matmul + on-chip gated epilogue fused into
    one kernel (the fused-node codegen reference).
  - `chunk_h_scan/` — **T2**: a sequential cross-chunk recurrence kept in one
    kernel, with the state resident as both a matmul operand and a Vec accumulator
    (the resident-state fusion reference).
  - `deltanet_chunk/` — **T3**: the opaque-node composition (triangular inverse via
    the `pto-kernels` rec_unroll kernel) and the full DeltaNet end-to-end forward.
    - `probe_triinv.py` — pins the opaque tri-inv kernel's input/output contract.
    - `delta_e2e.py` — the complete chunked DeltaNet forward composed on the
      substrate (kkt → solve_tril opaque → recompute W,U → cross-chunk scan →
      chunk_o), validated bit-faithfully against an fp32 reference. The worked
      example for the auto-fuser.

## Dependencies

`pto-fuser` finds `pto-einsum` as a sibling directory by default; override with the
`PTO_EINSUM` environment variable. The opaque tri-inv node reuses
[`pto-kernels`](../pto-kernels) (override with `PTO_KERNELS`). Runtime env, as for
the substrate:

```
export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
python prototypes/deltanet_chunk/delta_e2e.py --B 8 --H 32 --nc 64 --C 64
```
