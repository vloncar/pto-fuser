"""Graph-replay backend — M3, planner lever #1 (dispatch-elim).

`docs/FUSER_DESIGN.md` §4 lever 1 / §5 "Graph-replay": an NPUGraph captures the
staged chain's per-stage ctypes launches and replays them as a *single* dispatch.
This collapses N per-stage host launches into one — the win that lets a staged
pipeline *beat* a monolithic mega kernel in the launch-bound small-`T` regime
(~2.4× at one chunk) while staying perf-neutral and bit-exact when compute-bound.

Two things make a chain capturable, and this backend supplies both:

1. **No JIT inside the capture region.** The substrate's one-shot ``einsum()``
   rebuilds the kernel (codegen + dlopen + first-call workspace setup) on every
   call — all host work, none of it capturable. `CaptureExecutor` instead builds a
   *persistent* einsum runner per node once (honoring the planner's read-mode /
   fused-store annotations at build time) and reuses it, so the captured region is
   pure device launches.
2. **No per-call host sync.** The opaque tri_inv lowering syncs to order its raw
   launch against the torch stream; under capture those syncs are dropped via
   `registry.capture_mode()` (the launch is recorded on the capture stream in
   order). Mega is *not* capturable for the dual reason — its wrapper does a
   per-call ``cu.cpu()`` host read — which is precisely why staged-captured wins.

Usage::

    gr = GraphReplayExecutor().capture(program, bindings)
    out = gr.replay(new_bindings)          # one dispatch; bit-exact vs staged

`replay` copies the new operands into the static input buffers the graph was
captured over, replays, and returns the (stable, pool-resident) output tensors.
Shapes are fixed at capture time (design §9: static-shape assumption); a new shape
needs a new capture.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from . import registry as _registry
from .executor import StagedExecutor, _cast, substrate_modes
from .ir import EinsumNode, Program
from .registry import OpaqueRegistry


class CaptureExecutor(StagedExecutor):
    """A `StagedExecutor` whose einsum nodes use persistent, pre-built runners.

    Identical results to the staged backend (it *is* a StagedExecutor for glue,
    opaque, and tensor ops) but the einsum dispatch is a cached
    ``EinsumBuilder(...).build()`` runner rather than the rebuild-every-call
    ``einsum()`` convenience — the prerequisite for capturing the launch chain.
    """

    def __init__(self, registry: Optional[OpaqueRegistry] = None) -> None:
        super().__init__(registry)
        self._runners: Dict[tuple, object] = {}

    def _exec(self, node, env: Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(node, EinsumNode):
            a, b = (env[n] for n in node.inputs)
            runner = self._runner(node, a, b)
            return _cast(runner(a, b), node.out_dtype)
        return super()._exec(node, env)

    def _runner(self, node: EinsumNode, a: torch.Tensor, b: torch.Tensor):
        key = (node.equation, tuple(a.shape), tuple(b.shape), a.dtype,
               node.read_mode, node.fuse_out)
        runner = self._runners.get(key)
        if runner is None:
            from pto_einsum import EinsumBuilder       # noqa: import-after-path
            # The read-mode/fused-store toggles affect *codegen*, so they must be
            # set while build() runs, not at launch (cf. executor.substrate_modes).
            with substrate_modes(node.read_mode, node.fuse_out):
                runner = EinsumBuilder(
                    node.equation, [a.shape, b.shape], a.dtype).build()
            self._runners[key] = runner
        return runner


class GraphReplayExecutor:
    """Capture a `Program` into one NPUGraph; replay it as a single dispatch."""

    def __init__(self, registry: Optional[OpaqueRegistry] = None) -> None:
        self._exec = CaptureExecutor(registry)
        self._graph = None
        self._static_in: Dict[str, torch.Tensor] = {}
        self._out: Dict[str, torch.Tensor] = {}
        self.program: Optional[Program] = None

    def capture(self, program: Program, bindings: Dict[str, torch.Tensor],
                warmup: int = 3) -> "GraphReplayExecutor":
        missing = [n for n in program.inputs if n not in bindings]
        if missing:
            raise ValueError(f"missing bindings for inputs: {missing}")

        # Static input buffers the graph is captured over; replay writes into these.
        self._static_in = {k: bindings[k].clone() for k in program.inputs}

        # Warmup (eager, outside capture): builds every persistent runner, runs the
        # one-time workspace setup, and primes the opaque/JIT caches — so the
        # capture region that follows contains only device launches.
        env = None
        for _ in range(max(1, warmup)):
            env = self._exec.run(program, self._static_in, return_env=True)
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            with _registry.capture_mode():     # opaque nodes drop their host syncs
                env = self._exec.run(program, self._static_in, return_env=True)
        torch.npu.synchronize()

        self._graph = graph
        self.program = program
        self._out = {name: env[name] for name in program.outputs}
        return self

    def replay(self, bindings: Dict[str, torch.Tensor],
               clone: bool = True) -> Dict[str, torch.Tensor]:
        """Replay the captured graph on `bindings` (must match captured shapes).

        Returns the output tensors. They are the graph's pool-resident buffers, so
        `clone=True` (default) hands back independent copies that survive the next
        replay; pass `clone=False` in a tight measurement loop to skip the copy.
        """
        if self._graph is None:
            raise RuntimeError("capture() must be called before replay()")
        for name, buf in self._static_in.items():
            src = bindings[name]
            if src.shape != buf.shape:
                raise ValueError(
                    f"replay shape {tuple(src.shape)} for {name!r} != captured "
                    f"{tuple(buf.shape)} (a new shape needs a new capture)")
            buf.copy_(src)
        self._graph.replay()
        if clone:
            return {k: v.clone() for k, v in self._out.items()}
        return dict(self._out)
