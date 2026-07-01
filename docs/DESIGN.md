# pto-fuser — design

The **fusion layer** over the [`pto-einsum`](../../pto-einsum) contraction library.
`pto-einsum` compiles one tensor contraction at a time to Ascend Cube/Vec kernels and
is *soft-frozen*; `pto-fuser` consumes it as a pinned dependency and owns everything
*graph-level*: representing a sequence of einsum-core stages + opaque foreign kernels +
Vec glue as an IR, and lowering that IR to an efficient execution plan.

This is the single source-of-truth design document. It states the thesis, the IR, the
**separated compilation stack** (transforms / cost model / policy / verification), the
execution backends, the correctness discipline, the transform library, the worked
forwards, and the roadmap toward generating megakernels. It reflects the realized state
of the repository; historical design iterations live in the git history, not here.

---

## 1. Thesis

Three claims, each validated by prototype or benchmark, define the bet:

1. **Fusion is a separate layer on top of `einsum()`, not baked into it.** The library
   compiles *one* contraction and is soft-frozen. Everything graph-level — which read
   mode each stage uses, which stores are fused, which stages collapse into a single
   resident-state kernel, when to capture a dispatch graph — lives here.

2. **The chunk-attention taxonomy decomposes into library primitives plus a small set
   of opaque nodes.** Validated end-to-end for DeltaNet, GDN, and KDA: matmul-core
   contractions + Vec glue + a single opaque triangular-inverse node reproduce the full
   gated forwards bit-faithfully against an fp32 reference. No missing primitive — this
   is integration, not new capability.

3. **Fusion buys dispatch-elimination and on-chip residency, not raw compute.** At the
   production GDN shape the torch *glue* (cumsum, gating, masking, scaling) is 66–87% of
   each stage and is HBM-bandwidth-bound; the contraction is the small part. So the wins
   are (a) collapsing N per-stage dispatches into one graph replay, and (b) keeping an
   intermediate (the scan state `S`, the qk matrix) on-chip across a boundary the staged
   lowering round-trips HBM.

The consequence: the fuser is a **scheduler + rewriter**, not (yet) a megakernel
generator. §8 maps the controlled path from here to a megakernel-producing system.

---

## 2. Intermediate representation (`ir.py`)

A `Program` is an ordered list of typed nodes over a name→tensor environment (the
unrolled static chain — exactly what graph capture wants). Three **compute** node types
— the taxonomy proved these suffice — plus one host-plumbing type:

| node | role |
|------|------|
| `EinsumNode` | one library contraction. Carries the selection annotations `read_mode` (`NN`\|`auto`) and `fuse_out`, which are **transform outputs**, not user input. |
| `OpaqueNode` | a foreign hand-optimized kernel the matmul-core cannot express (today: the `tri_inv_rec_unroll` triangular inverse). Hosted by registry key with a pinned dtype/layout contract. |
| `VecGlueNode` | a standalone Vector op — `tril` / `add` / `sub` / `mul` / `scale`. Accumulates in fp32, casts via `out_dtype`. |
| `FusedNode` | a sub-chain collapsed into ONE hosted kernel (resident-state scan, gated-qk). The only multi-output node (a scan emits both per-chunk readouts and the final state). |
| `TensorOp` | host plumbing (`reshape`/`contiguous`/`transpose`/`permute`/`cast`/`slice`/`stack`/`zeros`) — **not** a compute type; it threads intermediates between kernels and collapses to buffer-binding metadata under capture. |

`Program.__post_init__` does a use-before-def / undeclared-output check, so a malformed
rewrite fails at construction.

**Dtype is part of the opaque contract.** The single hardest historical e2e bug was
handing the fp16-only tri-inv kernel fp32 bytes → deterministic NaN. The registry
lowering inserts the cast explicitly; there is no implicit dtype agreement.

---

## 3. The separated compilation stack

The core architectural principle: **the transformation, the heuristic, and the
verification are three separable pieces**, not one gated build-flag branch. Adding an
optimization means adding a pure rewrite + a cost prediction, never threading a flag
through the builder, the executor, and every benchmark.

```
canonical Program                        (§3.1  always-valid NN baseline)
      │
      ▼   Policy.pipeline(program, Features)      (§3.3  the heuristic: which/what order)
      │      └── CostModel.predict(...)           (§3.4  seeded benefit ranks)
      ▼
  for each proposed Transform:            (§3.2  pure Program→Program rewrites)
      cand = transform.apply(prog)
      verify cand vs canonical  ──────────(§3.5  frob + determinism + faster)
      keep iff gated-green ∧ deterministic ∧ faster
      ▼
  lowered Program + CompilationReport     (§3.6  provenance: predicted vs measured)
```

Everything below `compile_program` is deterministic and pure except the verifier, which
is the only part that touches the device.

### 3.1 Canonical form (`canonicalize`)

Every `EinsumNode` forced to the Phase-A **NN** read with **no** fused store — the
lowering the `StagedExecutor` honors unconditionally and the correctness reference the
verifier gates against. The builders (`build_gdn_program`, `build_kda_program`,
`forwards.build_deltanet_program`) emit exactly this all-staged form: every stage its
own einsum/glue node, the cross-chunk scan unrolled over chunks, no fused nodes. Every
optimization is then an *optimization proven against this floor*, never the default.

### 3.2 Transforms (`transform.py`, `transforms/`)

A `Transform` is a pure `Program → Program` rewrite with a kebab-case `name`, a
one-line `summary`, a `match(program) → int` predicate (number of rewritable sites,
`0` = not applicable), and an `apply(program) → TransformResult`. It measures nothing,
decides nothing, touches no device — it only rewrites the IR where its pattern occurs.
Structural transforms are parameterized by the forward's dims (the pass "options",
mirroring the `cce-mlir` pass shape). The transform library is §6.

### 3.3 Policy (`policy.py`)

The one place that *chooses*. Given the canonical program and its `Features`, it
instantiates every candidate transform (parameterized by the dims), drops those that do
not structurally match this program (so it stays forward-agnostic — GDN vs KDA vs a flat
family is decided by `match`, not by the policy), asks the cost model whether each is
worth trying, and orders the survivors: **structural fusions first** (they remove
einsums), then the read-mode / fused-store annotation levers over whatever einsums
remain. It returns *proposals*, not decisions.

### 3.4 Cost model (`cost.py`)

Predicts *whether a transform is likely to pay* from the shape/dtype `Features`
(`B,H,nc,C,D,dtype`; derived `T`, `M`, `N`, `regime`). It ranks and prunes; it never
decides alone. The predictions are **seeded from the measured GDN scaling grid**: the
fuser is launch/dispatch-bound at small head count and bandwidth-bound at large head
count, and the fuser-wins crossover context length grows as heads shrink. So
resident-state benefit rises with chunk count, glue-absorption benefit rises with head
count, and the v2 (L2-ring) fused kernels are shape-gated on the score·head product.
The ranks are coarse by design — the verifier is ground truth; the model exists to order
the pipeline and to be **recalibrated against report data** (predicted vs measured).

### 3.5 Verification (`compile.py`, reusing `gate.py` / `fusion.py`)

Each proposed rewrite is run on device and kept **iff** it is:
- **gated-green** — outputs frob-equal to the **canonical** reference (never to the
  previous candidate; the correct staged lowering is always the floor). Ungated "wins"
  were the historical trap — a broken zero/garbage pipeline looks fast.
- **deterministic** — the candidate run twice is bit-identical. This is the guard that
  caught the historical mega H=64 non-determinism (a missing cumsum Vec→Scalar edge) a
  single-run frob check masks; mandatory on any fused/scan lowering.
- **faster** — measured against the current best; on keep, the best is updated so each
  subsequent transform must improve on the accumulated result.

A mispredicted win is therefore a perf event, never a correctness one — worst case the
canonical staged lowering stands.

### 3.6 Report (`report.py`)

Every proposed transform leaves a `TransformRecord`: predicted benefit + rationale,
whether it matched, and — if verified — the measured frob / determinism / speedup that
decided it. The `CompilationReport` is both the human-readable trace of *why the lowered
program looks the way it does* and the data that recalibrates the cost model.

Off device (or `verify=False`) the loop still runs the policy and applies the
transforms, producing the lowered program and an *unverified* report — which is what the
structural unit tests exercise, and what makes the whole heuristic/transform layer
testable without an NPU.

---

## 4. Execution backends

- **Staged (`StagedExecutor`)** — each node → its own library `.so` (persistent
  workspace; setup/exec/teardown), stages share GM tensors. Always available; the
  correctness reference for every other backend.
- **Graph-replay (`GraphReplayExecutor` / `CaptureExecutor`)** — the staged chain
  captured into one `NPUGraph` and replayed as a single dispatch. Two prerequisites,
  both supplied: no JIT inside capture (persistent pre-built einsum runners, keyed
  including the read-mode/fuse-out annotations) and no per-call host sync (the opaque
  tri-inv drops its two stream syncs inside the capture region via
  `registry.capture_mode()`). The default production backend for static-shape chains; a
  replay whose shape differs from capture raises (static-shape enforced, not silently
  wrong).
- **Fused-node (`fused.py`)** — hosts a proven hand-written kernel (resident-state
  `chunk_h_scan`, gated `kkt_gated_native` / `gated_qk_native` / `qk_prologue`) as a
  persistent, sync-free `FusedKernel` so a `FusedNode` is itself graph-capturable. Each
  kernel is its **own `.so` sharing GM** with the surrounding stages — the form that
  works today. Inlining an opaque AICORE device-fn into one `.so` with the library
  matmul core is left as research (§8, §9).

The library stays the single source of the matmul/Vec/transpose codegen. The fuser
composes, captures, and hosts; it does not fork the core.

---

## 5. Correctness gating (non-negotiable)

`gate.py`: `frob_rel` (relative Frobenius norm under a fixed tolerance, 2e-2 per stage,
tighter for pure-matmul) and `gate_determinism` (twice, bit-identical). `fusion.decide`
composes them with a back-to-back measurement into a single keep/drop verdict, and the
compile driver applies exactly this to every transform. The gate is part of the compile
loop, not a separate test phase: a transform that does not pass is discarded and the
program without it stands.

---

## 6. The transform library

### Universal (annotation levers over any `EinsumNode`)

- **`enable-direct-reads`** — `read_mode` NN → `auto`, letting the library pick its
  direct-read mode (NT / NN-strided / TN) instead of Phase-A. Huge on the head-strided
  GDN/KDA family (the head axis `h` is a non-innermost batch axis, so Phase-A pays a
  strided gather the direct read removes — measured **2.7–10.3×** on the GDN stages),
  ~1.0× on flat `[M,C,D]` families. The library auto-selects *which* mode fires; the
  transform only proposes direct-read-vs-Phase-A, and the verifier keeps it per its
  measured, layout-dependent payoff.
- **`enable-fused-store`** — `fuse_out` False → True, permitting the operand-swap that
  exposes the fused permuted store when free1 is innermost (drops Phase C; 3.6–3.9× on
  `->bsht`, 2.11× when the swap fires).

### Structural (forward-shaped fusions, `transforms/gdn.py` + `transforms/kda.py`)

- **`lower-resident-scan`** (GDN) / **`lower-perdim-scan`** (KDA) — replace the unrolled
  `nc`-chunk cross-chunk scan (which writes the carried state `S` back to HBM every
  chunk) with the `chunk_h_scan` `FusedNode` that keeps `S` resident on-chip, plus the
  parallel `v_new = U − W·h_out` recompute. Removes the per-chunk HBM round-trip of `S` —
  a bandwidth win graph capture cannot touch, so it fuses **broadly** (4.40× at B1H4nc8,
  2.28× at B8H32nc16), not just launch-bound. KDA differs only in a per-dimension decay
  vector (`perdim_decay=True`, an `[B,H,nc,D]` operand).
- **`absorb-gated-kkt`** / **`absorb-gated-chunk-o`** (GDN) — match the
  `(a·bᵀ) einsum → mul(coef) → tril? → contiguous` region and replace it with a native
  gated-qk `FusedNode` (matmul-core + on-chip gate/mask epilogue) so the qk matrix never
  lands in HBM. The scalar-gated epilogue is shared across the family (vanilla LA,
  RetNet, Mamba-2, GDN all differ only in the host `g`/`β`). The native `[M,C,D]`
  variants read the program's own batch with no transpose bridge; the v2 kernels
  additionally keep the qk tile L2-resident (no `[M,C,C]` round-trip) — measured kkt
  1.18–1.20×.
- **`absorb-qk-prologue`** (KDA) — the per-**channel** analog: KDA's gate is baked into
  the operands (`exp(±g)`), so the decay folds into the matmul *load* (`q⊙exp(g)`,
  `k⊙exp(−g)`) rather than a scalar score factor, and the causal mask rides in the
  kernel's Vec pass. `q_eff` is left in place (it also feeds `o_inter`).

Each structural transform matches only its own family (a GDN-vs-KDA discriminator on the
program's declared coefficient inputs and the self-outer-product vs prescaled-operand
einsum shape), is idempotent (it does not re-fire on an already-fused program), and its
rewrite is proven byte-identical to the previously hand-written fused lowering.

---

## 7. Worked forwards

| forward | where | notes |
|---------|-------|-------|
| DeltaNet | `forwards/deltanet.py` | the 5-stage delta-rule backbone (gate g=0); the reference the design was first calibrated to, bit-faithful at M=16384. |
| GDN | `examples/attention/_gdn_full.py` | full gated forward, scalar decay; canonical builder + the three GDN transforms. `gdn_reference` == RefGDN to fp64. |
| KDA | `examples/attention/_kda_full.py` | same equations, per-dim gate baked into operands; canonical builder + the two KDA transforms. |

All gating is **host-precomputed coefficient tensors** multiplied into the operands /
scores by `mul` glue (the cumsum/`exp` arithmetic a future on-device glue feature would
fold in is done on the host), proven bit-equal to the fp32 reference. `compile_program`
lowers each canonical forward and gates the result end-to-end; the GDN/KDA `benchmarks/`
put the compiled forward head-to-head against the megagdn/megakda megakernel (the
megakernel is a **benchmark baseline only** — it is not in the decision pipeline; the
fuser calls only pto-einsum, Vec glue, its own hosted fused kernels, and the opaque
tri-inv).

### Measured regime (the calibration data)

- **Read modes** — 2.7–10.3× on the head-strided GDN stages; ~1.0× flat DeltaNet.
- **Graph capture** — 2.4–4.0× launch-bound (small batch / few large chunks),
  perf-neutral and bit-exact when compute-bound (T≥512).
- **Resident scan** — 4.40× / 2.28× (fuses broadly; removes the S round-trip).
- **Glue absorption** — native kkt 1.96×; native+v2 kkt 1.18–1.20×, chunk_o to parity.
- **GDN vs mega across H×T** — the fuser-wins crossover context length grows as heads
  shrink (H1 wins to ~32k, H2 ~16k, … H16 ~2k, H64 never): small-H is launch/dispatch
  territory (fuser), large-H long-T is bandwidth territory (mega). This is exactly the
  regime split the cost model encodes.

---

## 8. Roadmap toward a megakernel-producing system

The separated stack is deliberately shaped so that megakernel generation is *one more
transform* — the hardest, added last, inside a proven envelope. The controlled steps,
each gated by the §5 discipline and each new fused pattern hand-proven by a prototype
**before** it is generalized into a generator (the generator never emits an unproven
pattern):

1. **Fusion-region identification** — a transform that *annotates* maximal einsum+glue
   producer-consumer chains where the intermediate is the dominant HBM traffic, and
   estimates the L2-residency win. Analysis only, no codegen; validated by having it
   rediscover the regions the current structural transforms already fuse.
2. **Template emission for the hardware-clean class** — generate exactly the pattern the
   Ascend a2/a3 hardware supports without landmines: matmul-core + per-channel (1D)
   fixpipe epilogue (the scalar-gate absorption that *is* fixpipe-expressible),
   parameterized from a region, every emission gated by the verifier. Deliberately
   excludes 2D-mask-in-fixpipe and speculative on-chip Cube↔Vec datapaths (the Cube↔Vec
   exchange is physically GM-only on this hardware).
3. **Widen the template class one proven pattern at a time** — per-tile FFTS interleave,
   then cube+vec mix kernels, then 2D-mask Vec passes — each hand-prototyped and gated
   first, then folded into the generator. Mix kernels with FFTS come only after the
   single-unit templates are solid.

**Deferred (not started; the eventual end goal).** Integration into the `cce-mlir`
compiler stack, which operates on the lower-level CCE intrinsics below the pto tile
level. This design borrows its naming/pass conventions (a named pass with a summary,
options, and an opt-in match predicate) so the transforms map cleanly onto compiler
passes later, but the fuser does not touch that stack now, and the MLIR rewrite is out
of scope until the work grows past the linear-attention vehicle into a general
tensor-compiler product.

---

## 9. Risks / open items

- **Static-shape assumption for graph capture.** Replay needs fixed shapes; dynamic
  seqlen / chunk count needs a family of captures keyed by shape bucket (single-shape
  capture is what is built; bucketing is deferred).
- **Opaque-node single-`.so` inlining.** Reconciling the opaque AICORE kernel's build
  flags with the library's inside one `.so` is unproven; staged multi-`.so` sharing GM
  works today and is what is used.
- **The Cube↔Vec GM wall.** On a2/a3 the Cube and Vec units exchange data only through
  GM; the fixpipe eltwise is per-channel 1D (can express a scalar/per-channel gate but
  not a 2D triangular mask). This bounds what §8 can generate as a single kernel and is
  why the roadmap emits the 1D-clean class first.
- **`tri_inv` chunk-size ceiling.** The opaque `tri_inv_rec_unroll` supports
  `C ∈ {16,32,64,128}` and silently returns garbage (and can poison the aicore) at
  C=256; a larger chunk needs a new tri-inv (out of scope, untouched here).
- **Library gaps** surfaced by the fuser are logged to the `pto-einsum` roadmap as
  demand-driven items, made behind a default-off flag, not patched around in the fuser.
