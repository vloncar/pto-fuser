# pto-fuser

The **fusion layer** on top of the [`pto-einsum`](../pto-einsum) contraction
**library**. `pto-einsum` compiles individual tensor contractions to Ascend Cube/Vec
kernels and is *soft-frozen*; `pto-fuser` consumes it as a pinned dependency and is
where the *graph-level* work lives: scheduling sequences of einsum-core stages, opaque
foreign nodes, and Vec glue into efficient kernels.

The guiding thesis (validated across the chunk-attention taxonomy): **fusion is a
separate layer on top of the einsum primitives, not baked into `einsum()`.** The
reusable unit is a tile-matmul core with pluggable load front-ends + epilogue, plus a
registry of opaque hand-optimized nodes (e.g. triangular inverse) that the graph can
host but that are not the matmul-core shape.

## Layout

- `docs/FUSER_DESIGN.md` — the design (thesis, IR, optimization features, codegen
  targets, correctness gating, build status).
- `docs/IMPLEMENTATION.md` — what is actually built, tracked against the design. The
  two are kept in sync: the design states the target, the implementation doc records
  the realized state per feature.
- `src/pto_fuser/` — the package: the IR (`ir.py`), the staged executor (`executor.py`),
  the opaque-kernel registry (`registry.py`), the correctness gates (`gate.py`), the
  read-mode / fused-store **Planner** (`planner.py`), the **graph-replay backend**
  (`graph.py` — capture the staged chain, replay as one dispatch), the **fused-node
  backend + fusion decision** (`fused.py` / `fusion.py` — host the resident-state scan
  and gated-kkt kernels as single-dispatch nodes, kept only where they gate bit-faithful
  + deterministic AND beat staged-captured), the hosted device kernels (`kernels/`), and
  the reference forwards (`forwards/` — DeltaNet, GDN stages, the fused-stage
  head-to-heads).
- `examples/` — runnable demonstrations: a minimal program, the chunked-attention zoo
  (vanilla LA, RetNet, GLA, Mamba-2, GDN, KDA), one demo per feature (`workflow/`), and
  the per-feature GDN benchmark (`benchmarks/`). See `examples/README.md`.
- `tests/` — off-NPU structural/decision/decomposition tests and NPU forward/feature
  tests. The `*_npu.py` device tests are skipped unless `PTO_RUN_NPU=1`, so the default
  `pytest` is the device-free suite.

## Dependencies

`pto-fuser` finds `pto-einsum` as a sibling directory by default; override with the
`PTO_EINSUM` environment variable. The opaque tri-inv node reuses
[`pto-kernels`](../pto-kernels) (override with `PTO_KERNELS`). Runtime env, as for the
library:

```
export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
python examples/minimal.py
python examples/attention/gdn.py
python examples/benchmarks/gdn_features.py --B 1 --H 4 --nc 8
```
