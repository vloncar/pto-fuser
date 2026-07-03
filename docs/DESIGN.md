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

## Pipeline at a glance

The path from a forward's definition to the executed device chain. Each band is
detailed by the section noted on the right; the crucial property is that a transform
never emits source — it swaps a subgraph of primitive nodes for a node that *names* an
already-proven kernel, and graph capture folds the resulting chain into one dispatch.

```
   build_*_program(dims)  ──▶  Program = [ EinsumNode | VecGlueNode | OpaqueNode        §2  IR
                                          | FusedNode | TensorOp ] + inputs/outputs
                                             │
       ┌─────────────────────────────────────┼─────────────────────────────────────┐
       │  compile_program(program, Features, bindings)                              │  §3  separated
       │     canonicalize ─▶ Policy/CostModel propose ─▶ verify (frob+det+faster)   │      compilation
       │                                     │            ─▶ keep / roll back        │      stack
       └─────────────────────────────────────┼─────────────────────────────────────┘
                                             ▼
                    lowered Program  +  CompilationReport (predicted vs measured)     §3.6
                                             │
                          GraphReplayExecutor.capture(...) ─▶ ONE NPUGraph replay     §4  backends
                                             │      (the "emitted code": a captured
                                             │       device launch-chain, not text)
                                             ▼           per-node dispatch
              EinsumNode  ─▶  pto-einsum library          (matmul core; §4)
              FusedNode   ─▶  FusedKernelRegistry `.so`    (hosted proven kernel; §4, §6)
              OpaqueNode  ─▶  OpaqueRegistry               (tri_inv; §2 dtype contract)
              VecGlueNode ─▶  torch Vec op                 (mul/add/sub/scale/tril-mask)
              TensorOp    ─▶  torch reshape/slice/…        (host plumbing, folds under capture)
```

The middle band — canonicalize → propose → verify → dispose — is the heart of the
design and is expanded on its own in §3; the node types it rewrites are §2, the
backends that execute the result are §4, and the transforms that do the rewriting are
§6. The two dispatch paths that matter for the eventual `cce-mlir` port: an
`EinsumNode` always goes to the frozen library, a `FusedNode` always names a
pre-proven `.so` — new device capability enters through neither a transform nor a
policy but only by first hand-proving a kernel (§8).

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

Every `EinsumNode` forced to the input-transpose **NN** read with **no** fused store — the
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
loop, not a separate testing step: a transform that does not pass is discarded and the
program without it stands.

---

## 6. The transform library

### Universal (annotation levers over any `EinsumNode`)

- **`enable-direct-reads`** — `read_mode` NN → `auto`, letting the library pick its
  direct-read mode (NT / NN-strided / TN) instead of the input transpose. Huge on the head-strided
  GDN/KDA family (the head axis `h` is a non-innermost batch axis, so the input transpose pays a
  strided gather the direct read removes — measured **2.7–10.3×** on the GDN stages),
  ~1.0× on flat `[M,C,D]` families. The library auto-selects *which* mode fires; the
  transform only proposes direct-read-vs-input-transpose, and the verifier keeps it per its
  measured, layout-dependent payoff.
- **`enable-fused-store`** — `fuse_out` False → True, permitting the operand-swap that
  exposes the fused permuted store when free1 is innermost (drops the output transpose; 3.6–3.9× on
  `->bsht`, 2.11× when the swap fires).

### Structural — resident-state scan (`transforms/gdn.py` + `transforms/kda.py`)

- **`lower-resident-scan`** (GDN) / **`lower-perdim-scan`** (KDA) — replace the unrolled
  `nc`-chunk cross-chunk scan (which writes the carried state `S` back to HBM every
  chunk) with the `chunk_h_scan` `FusedNode` that keeps `S` resident on-chip, plus the
  parallel `v_new = U − W·h_out` recompute. Removes the per-chunk HBM round-trip of `S` —
  a bandwidth win graph capture cannot touch, so it fuses **broadly** (4.40× at B1H4nc8,
  2.28× at B8H32nc16), not just launch-bound. KDA differs only in a per-dimension decay
  vector (`perdim_decay=True`, an `[B,H,nc,D]` operand). Each matches only its own family
  and is idempotent; the rewrite is byte-identical to the hand-written fused lowering.
- **`batch-chunk-intra-score`** (`transforms/chunked.py`) — the linear family
  (vanilla/RetNet/Mamba-2/GLA) unrolls its intra-chunk score per chunk; this collapses all
  `nc` per-chunk `tril(q̃·k̂ᵀ)` scores into ONE batched proven kernel over `M = N·nc`,
  reusing `gated_qk_native_v2` (scalar gate) or `qk_prologue` (GLA per-channel). The gate
  kind is a forward-declared option (`Features.per_dim_gate`). This is the §8 B2 widening —
  more coverage for the proven kernels, no new device code.
- **`fuse-chunk-o-flash`** (`template.py`) — the score→output "flash" fusion, and the
  first **genuinely-new device kernel** the roadmap allows (`kernels/qkv_fused.h`,
  `qkv_flash_native` / `_v2`). It folds the whole scalar-gate chunk_o — `q·kᵀ → gate/mask
  → ·v` (Aqk einsum + `coef_o` mul + contiguous + the `o_intra` `A·v` contraction) — into
  one Cube→Vec→Cube launch, recomputing the gate + causal mask on-chip, so the `[M,C,C]`
  masked score never lands in HBM. The emitted kernel is the **double-buffered interleave
  V2**: a two-slot per-core L2 ring where the Cube produces the *next* tile's score while
  the Vec masks the current one, hiding the Cube↔Vec FFTS handshake, so **S and A are both
  L2-resident** (the V1 two-pass materializes them and pays a round-trip — it is the
  bit-exact oracle, not the emitted kernel). Hand-proved bit-exact (V2 bit-identical to
  V1), and a **kept win under graph capture at large head count** (`H∈{16..64}`, ≈3–5%
  end-to-end on GDN — the whole score→output slice) while the verifier **disposes it at
  small H** (few tiles to overlap; capture already covers the dispatch). It is a
  **standalone** transform, *not* part of the epilogue generator, so its per-shape
  keep/drop never disturbs the kkt fusion (bundling it once dropped the proven kkt fusion
  along with it — §8).
- **`fuse-perdim-chunk-o-flash`** (`template.py`) — the **per-channel-gate twin** of the
  scalar flash, extending the score→output fusion to the KDA/GLA family
  (`kernels/qkv_prologue_fused.h`, `qkvp_flash_native_v2`). The per-dim decay rides on the
  *operands* (a Vec prescale `q⊙coef_ag`, `k⊙coef_bg`) rather than a scalar epilogue, so
  the pipeline gains a leading stage: **Vec prescale → Cube score → Vec tril → Cube A·v**,
  a four-stage double-buffered ring with ops, S *and* A all L2-resident. It matches the
  canonical per-dim chunk_o (the `q_eff`/`k_eff` prescale muls + Aqk + tril + contiguous +
  `o_intra`), keeps `q_eff` (it also feeds `o_inter`), and reads `(qF, kF, v, coef_ag,
  coef_bg)`. Reuses `prologue_fused`'s `prescale_one`/`mask_one` verbatim and the same two
  matmul configs as the scalar flash — the only new device code is the 4-stage
  choreography. Bit-exact (V2 bit-identical to V1) and a kept win under capture at large H
  (KDA H16nc8 ≈2.7%, H32nc8 ≈3% end-to-end); disposed at small H, like the scalar flash.

### Structural — contraction+epilogue template emission (`template.py`, §8)

- **`fuse-contraction-epilogue`** — the region-driven generator. For each contraction it
  extracts the local **epilogue unit** (einsum + downstream single-consumer glue chain +
  per-operand prologue) and, if a **proven** `Template` matches, rewrites the unit into
  that template's `FusedNode`. The template registry hosts the hand-proved gated-matmul
  kernels: `kkt-gated-native` (`A = tril((k·kᵀ)⊙coef,-1)`), `chunk-o-gated-native`
  (`(q·kᵀ)⊙coef`, causal), and `qk-prologue` (KDA per-dim gate folded into the matmul
  load). The native `[M,C,D]` variants read the program's own batch (no transpose
  bridge); the v2 kernels keep the qk tile L2-resident (no `[M,C,C]` round-trip). One
  generator replaces three bespoke transforms — it is byte-identical to them (golden
  test) — and **never emits a pattern without a proven kernel**: a contraction whose
  epilogue matches no template is left staged (`epilogue_report` says which and why).

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

1. **Fusion-region identification** — **realized** (`analysis.py`,
   `identify_fusion_regions`). An *analysis* (not a rewrite) that partitions the
   canonical program into maximal fusible scopes cut at opaque-kernel boundaries, sizes
   every intermediate by a pure meta-tensor shape walk (no device), and scores each
   region by the internal HBM traffic fusing it would keep on-chip (the L2-residency
   opportunity — e.g. GDN splits into a 4-node contraction-epilogue region + a ~126-node
   recurrence region across the tri_inv cut, ~151 MB fusible on-chip). Emits nothing.
   Validated (`test_analysis.py`) that it rediscovers the regions the structural
   transforms already fuse and that a key safety property holds: **every structural
   transform stays inside a single region — a fusion never spans the tri_inv boundary.**
   The region representation (named analysis; explicit boundary-in / boundary-out /
   internal sets) is deliberately compiler-shaped so it maps onto a `cce-mlir`
   fusion/outlining analysis when this stack is reimplemented there.
2. **Template emission from a region** — **realized** (`template.py`,
   `FuseContractionEpilogue`). For each contraction the generator extracts its epilogue
   unit and, if a **proven** kernel template matches, emits that `FusedNode` — one
   region-driven generator replacing the three bespoke gated-fusion transforms
   (byte-identical to them; `test_template.py`), **never emitting a pattern without a
   proven kernel behind it**. `epilogue_report` classifies every contraction's epilogue.

   *Evidence that reshaped this step:* §8 originally named the per-channel (1D) *fixpipe*
   scale epilogue as the first class to generate (it is the one a matmul store can absorb
   without a Vec pass). But `pto-einsum` exposes no matmul+scale epilogue (a scaled
   matmul is einsum + a Vec op today), and — measured by `epilogue_report` on the real
   forwards — **no GDN/KDA stage is a pure fixpipe-1D scale**: their epilogues are 2D
   score masks and per-row scales, which the *Vec*-epilogue templates already cover. So
   the proven template family we generalize is matmul-core + on-chip Vec/mask epilogue;
   a fixpipe-1D kernel would have no consumer and is deferred (hand-prove it when a
   consumer appears, or expose it as a `pto-einsum` core capability) rather than built
   speculatively. This keeps the discipline: the generator only ever emits proven kernels.
   Still excluded, by the physical Cube↔Vec GM-only constraint: 2D-mask-in-fixpipe and
   speculative on-chip Cube↔Vec datapaths.
3. **Widen the template class one proven pattern at a time** — **first widening realized**
   (`transforms/chunked.py`, `BatchChunkIntraScore`). The linear family (vanilla LA,
   RetNet, Mamba-2, GLA) unrolls its intra-chunk score per chunk; this transform collapses
   all `nc` per-chunk scores into ONE batched kernel over `M = N·nc`, **reusing the proven
   kernels** the GDN/KDA templates already use — `gated_qk_native_v2` for a scalar gate,
   `qk_prologue` for GLA's per-channel gate — so the whole linear family now compiles
   through the generator with **no new device code** (byte-identical to the hand-written
   `fused_intra` lowering; NPU-verified RetNet + GLA). The gate kind (scalar vs per-dim) is
   the one property the canonical IR cannot carry, so it is a forward-declared compile
   option (`Features.per_dim_gate`) — a concrete signal for the eventual frontend. Further
   widenings (cube+vec mix kernels, then a genuinely new device kernel such as the deferred
   fixpipe-1D scale) follow the same rule: hand-prove the pattern, then fold it in — the
   generator never emits an unproven kernel.
4. **The first genuinely-new device kernel — chunk_o score→output "flash"** —
   **realized** (`kernels/qkv_fused.h`, `qkv_flash_native`; the `fuse-chunk-o-flash`
   transform in `template.py`). The residual-staging analysis showed every forward still
   stages `o_intra = A·v` after B1/B2: the fused qk kernel writes the masked score `A`
   `[M,C,C]` to HBM and a separate matmul reads it back. This kernel folds that second
   contraction in — `q·kᵀ → gate/mask → ·v` as one Cube→Vec→Cube launch, reusing the
   proven gated-qk epilogue (`kkt_epilogue_one`) between two matmul-core passes and adding
   only a plain-NN output config. Hand-proved **bit-exact** (two-pass V1, and the
   L2-resident interleaved V2 which is bit-identical — the 3-stage FFTS choreography is
   exact) and **≈5× faster than the staged path un-captured**.

   *Two things this step taught.* **(a) Standalone, independently-verified transform —
   never bundle.** An early version emitted flash *inside* the epilogue generator; under
   capture flash-V1 trades the small A round-trip for a larger S round-trip, so it is a
   dispatch-regime win that ≈regresses captured, and the bundled generator's keep/drop is
   atomic — the flash regression dragged the *proven kkt fusion* below the keep threshold
   and the verifier dropped **both**. Split into its own transform, propose/verify/dispose
   gates flash on its own merits with **zero collateral** to kkt. **(b) The captured win
   needed cross-tile overlap, and it landed.** The sequential single-slot V2 was correct
   and bit-identical but sync-bound (slower than V1). The emitted kernel is now the
   **double-buffered V2**: a two-slot per-core L2 ring where the Cube produces the next
   tile's score while the Vec masks the current one — the WAR hazards ordered for free by
   the Cube's own instruction stream (no FREE flags), so **S and A stay L2-resident** and
   the handshake is hidden. Result: flash V2 (still bit-identical to V1) is a **kept win
   under capture at large head count** (`H∈{16..64}`, ≈3–5% end-to-end on GDN), and the
   verifier disposes it at small H where there are too few tiles to overlap and capture
   already covers the dispatch. Exactly the separated stack working as designed — a kernel
   that wins in one regime and not another is a localized, per-shape perf decision, never
   a correctness event or a regression to a neighbor.

   *Then widened to the per-channel-gate family* (`fuse-perdim-chunk-o-flash`,
   `qkvp_flash_native_v2`): KDA/GLA put the decay on the operands, so the flash gains a
   leading Vec-prescale stage (Vec → Cube → Vec → Cube), a four-stage double-buffered
   ring — but it reuses the prologue's prescale/mask and the same matmul configs, so the
   only new device code is the choreography. Same discipline, same outcome: bit-exact,
   kept under capture at large H (KDA ≈3% e2e), disposed at small H. It also exercised the
   determinism gate for real — at H32nc8 the pre-existing intermittent scan FFTS race
   (~1/10, documented in the scan work) trips the gate on the combined candidate and the
   verifier falls back, even though the flash kernel itself is deterministic across 30
   isolated runs. The gate protecting correctness against a *neighbor's* latent race,
   without any change to the flash, is the discipline working end-to-end.

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
