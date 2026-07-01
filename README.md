# pto-fuser

The **fusion layer** on top of the [`pto-einsum`](../pto-einsum) contraction
**library**. `pto-einsum` compiles individual tensor contractions to Ascend Cube/Vec
kernels and is *soft-frozen*; `pto-fuser` consumes it as a pinned dependency and is
where the *graph-level* work lives: scheduling sequences of einsum-core stages, opaque
foreign nodes, and Vec glue into efficient kernels.

The guiding thesis (validated across the chunk-attention taxonomy): **fusion is a
separate layer on top of the einsum primitives, not baked into `einsum()`** — and within
that layer the **transformation, the heuristic, and the verification are three separable
pieces**. An optimization is a pure IR→IR *transform*; a *cost model* + *policy* decide
which to attempt; a measure-and-gate *verifier* keeps each only on a proven win. Adding a
lever is adding a rewrite + a cost prediction, not threading a build flag.

## Layout

- `docs/DESIGN.md` — the single design document: thesis, IR, the separated compilation
  stack (transforms / cost / policy / verification), backends, gating, the transform
  library, the worked forwards, and the roadmap toward megakernel generation.
- `src/pto_fuser/` — the package:
  - IR + backends: `ir.py`, `executor.py` (staged), `graph.py` (graph-replay),
    `fused.py` (hosted fused kernels), `registry.py` (opaque tri-inv), `gate.py`.
  - the separated stack: `transform.py` (canonical form + the universal read-mode /
    fused-store transforms) and `transforms/` (the forward-shaped resident-scan and
    glue-absorption rewrites); `cost.py` (the seeded cost model + `Features`);
    `policy.py` (program+features → ordered transform pipeline); `compile.py`
    (`compile_program` — the propose/verify/dispose driver); `report.py` (provenance).
  - `fusion.py` / `planner.py` — the verification primitives (`decide`, per-node
    read-mode measurement) the driver reuses; `kernels/` — the hosted device kernels;
    `forwards/` — the DeltaNet reference + fused-stage head-to-heads.
- `examples/` — runnable demonstrations: a minimal program, the chunked-attention zoo
  (vanilla LA, RetNet, GLA, Mamba-2, GDN, KDA), one demo per feature (`workflow/`), the
  full `compile_program` forward (`attention/gdn.py`), and the fuser-vs-megakernel
  benchmarks (`benchmarks/`). See `examples/README.md`.
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
