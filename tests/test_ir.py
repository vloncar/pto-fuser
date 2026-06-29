"""IR structural tests — no NPU required."""
import pytest
import torch

from pto_fuser import EinsumNode, Program, TensorOp, VecGlueNode
from pto_fuser.forwards import build_deltanet_program


def test_program_validates_dataflow():
    nodes = [
        EinsumNode("ij,jk->ik", ["a", "b"], "c"),
        VecGlueNode("scale", ["c"], "d", params={"scalar": 0.5}),
    ]
    prog = Program(nodes, inputs=["a", "b"], outputs=["d"])
    assert prog.outputs == ["d"]


def test_use_before_def_raises():
    nodes = [EinsumNode("ij,jk->ik", ["a", "missing"], "c")]
    with pytest.raises(ValueError, match="before it is produced"):
        Program(nodes, inputs=["a", "b"], outputs=["c"])


def test_undeclared_output_raises():
    nodes = [EinsumNode("ij,jk->ik", ["a", "b"], "c")]
    with pytest.raises(ValueError, match="never produced"):
        Program(nodes, inputs=["a", "b"], outputs=["nope"])


def test_deltanet_program_builds_and_is_well_formed():
    # nc small so the assertion is fast; structure is identical at any nc.
    prog = build_deltanet_program(B=1, H=2, nc=3, C=64, D=128, scale=0.1)
    assert prog.inputs == ["q", "k", "v", "beta"]
    assert set(["A", "T", "W", "U", "v_new", "h_state", "o"]).issubset(prog.outputs)
    # exactly one opaque node (the tri-inv), and einsum/glue counts scale with nc.
    from pto_fuser import OpaqueNode
    assert sum(isinstance(n, OpaqueNode) for n in prog.nodes) == 1
    assert sum(isinstance(n, EinsumNode) for n in prog.nodes) >= 5
