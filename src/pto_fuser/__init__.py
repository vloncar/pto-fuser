"""pto-fuser — the fusion / auto-fuser layer over the pto-einsum substrate.

M1 surface: the three-node IR, the staged executor, the opaque-kernel registry,
and the correctness gates. See docs/FUSER_DESIGN.md (design) and
docs/IMPLEMENTATION.md (realized state).
"""
from .ir import (EinsumNode, Node, OpaqueNode, Program, TensorOp, TensorRef,
                 VecGlueNode)
from .executor import StagedExecutor
from .registry import OpaqueContract, OpaqueRegistry, default_registry
from .gate import (GateResult, frob_rel, gate_determinism, gate_frob_rel,
                   gate_outputs)

__all__ = [
    "TensorRef", "Node", "EinsumNode", "OpaqueNode", "VecGlueNode", "TensorOp",
    "Program", "StagedExecutor", "OpaqueRegistry", "OpaqueContract",
    "default_registry", "GateResult", "frob_rel", "gate_frob_rel", "gate_outputs",
    "gate_determinism",
]
