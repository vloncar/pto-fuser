"""Planner keep-logic — off-NPU, with injected measurements (proven teeth).

The decision rule (design §8 M2) is: keep a lever only when it *fired* (the
substrate changed the lowering), the result is *gated-green* (frob_rel < tol vs the
baseline), AND it is *faster*. These tests script each of those three gates failing
in isolation and assert the verdict, with no NPU and no real substrate build.
"""
import torch

from pto_fuser import EinsumNode, Program, Planner
from pto_fuser.planner import _Measurement


def _const(val):
    return torch.full((4, 4), float(val))


class _ScriptedMeasure:
    """Returns a scripted _Measurement per (read_mode, fuse_out) variant.

    `script` maps ("NN",False)/("auto",False)/("NN",True) -> (t_us, out_val, in_nt,
    fusible, swapped). out_val builds a constant tensor so frob_rel is controllable.
    """
    def __init__(self, script):
        self.script = script

    def __call__(self, eq, a, b, *, read_mode, fuse_out):
        t, out_val, in_nt, fusible, swapped = self.script[(read_mode, fuse_out)]
        return _Measurement(t, _const(out_val), in_nt, fusible, swapped)


def _node():
    return EinsumNode("bihd,bjhd->bihj", ["a", "b"], "c", out_dtype=torch.float16)


def _eval(script, tol=2e-2):
    p = Planner(measure=_ScriptedMeasure(script), tol=tol)
    return p._evaluate(_node(), _const(0), _const(0))


def test_direct_read_kept_when_fired_gated_faster():
    v = _eval({
        ("NN", False): (100.0, 1.0, 0, 1, False),   # baseline
        ("auto", False): (40.0, 1.0, 1, 1, False),  # NT, identical, 2.5x faster
        ("NN", True): (100.0, 1.0, 0, 1, False),    # swap not fired
    })
    assert v["direct_read"].kept is True
    assert v["direct_read"].fired and v["direct_read"].gated_ok and v["direct_read"].faster
    assert v["operand_swap"].kept is False          # swapped=False -> never kept


def test_direct_read_dropped_when_slower():
    v = _eval({
        ("NN", False): (40.0, 1.0, 0, 1, False),
        ("auto", False): (90.0, 1.0, 1, 1, False),  # fired + gated but slower
        ("NN", True): (40.0, 1.0, 0, 1, False),
    })
    d = v["direct_read"]
    assert d.fired and d.gated_ok and d.faster is False and d.kept is False


def test_direct_read_dropped_when_gate_fails():
    v = _eval({
        ("NN", False): (100.0, 1.0, 0, 1, False),
        ("auto", False): (40.0, 9.0, 1, 1, False),  # fired + faster but garbage output
        ("NN", True): (100.0, 1.0, 0, 1, False),
    })
    d = v["direct_read"]
    assert d.fired and d.faster and d.gated_ok is False and d.kept is False


def test_operand_swap_kept_when_it_fires():
    v = _eval({
        ("NN", False): (100.0, 1.0, 0, 0, False),
        ("auto", False): (100.0, 1.0, 0, 0, False),  # direct read not eligible
        ("NN", True): (60.0, 1.0, 0, 1, True),       # swap fired, faster, gated
    })
    assert v["operand_swap"].kept is True
    assert v["direct_read"].kept is False


# -- plan() end-to-end off-NPU, with a fake executor -------------------------- #
class _FakeExecutor:
    def run(self, program, bindings, return_env=False):
        env = dict(bindings)
        for n in program.nodes:
            env[n.output] = torch.zeros(2, 2)
        return env if return_env else {o: env[o] for o in program.outputs}


def test_plan_annotates_and_dedups():
    # two EinsumNodes with the IDENTICAL shape class -> measured once, both annotated.
    nodes = [EinsumNode("bihd,bjhd->bihj", ["a", "b"], "c"),
             EinsumNode("bihd,bjhd->bihj", ["a", "b"], "d")]
    prog = Program(nodes=nodes, inputs=["a", "b"], outputs=["c", "d"])
    script = {
        ("NN", False): (100.0, 1.0, 0, 1, False),
        ("auto", False): (40.0, 1.0, 1, 1, False),
        ("NN", True): (100.0, 1.0, 0, 1, False),
    }
    p = Planner(executor=_FakeExecutor(), measure=_ScriptedMeasure(script))
    bindings = {"a": torch.zeros(1, 64, 16, 128), "b": torch.zeros(1, 64, 16, 128)}
    annotated, decisions = p.plan(prog, bindings)

    # dedup: one shape class -> 2 decisions (direct_read + operand_swap), not 4.
    assert len(decisions) == 2
    # both nodes pinned to the kept direct-read lowering.
    for n in annotated.nodes:
        assert n.read_mode == "auto" and n.fuse_out is False


def test_absorption_candidates_detects_glue_into_einsum():
    from pto_fuser import VecGlueNode
    nodes = [
        EinsumNode("nid,njd->nij", ["x", "y"], "raw"),
        VecGlueNode("tril", ["raw"], "masked", params={"diagonal": -1}),
        EinsumNode("nij,njd->nid", ["masked", "y"], "out"),
    ]
    prog = Program(nodes=nodes, inputs=["x", "y"], outputs=["out"])
    p = Planner(executor=_FakeExecutor())
    pairs = p.absorption_candidates(prog)
    assert ("masked", "out") in pairs
