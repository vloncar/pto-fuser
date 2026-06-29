"""pto-fuser — the fusion / auto-fuser layer over the pto-einsum substrate.

Surface: the three-node IR, the staged executor, the opaque-kernel registry, the
correctness gates (M1), the read-mode / fused-store Planner (M2), and the
graph-replay backend (M3, dispatch-elim). See docs/FUSER_DESIGN.md (design) and
docs/IMPLEMENTATION.md (realized state).
"""
from .ir import (EinsumNode, Node, OpaqueNode, Program, TensorOp, TensorRef,
                 VecGlueNode)
from .executor import StagedExecutor, substrate_modes
from .registry import (OpaqueContract, OpaqueRegistry, capture_mode,
                       default_registry)
from .gate import (GateResult, frob_rel, gate_determinism, gate_frob_rel,
                   gate_outputs)
from .planner import LeverDecision, Planner, format_decisions
from .graph import CaptureExecutor, GraphReplayExecutor

__all__ = [
    "TensorRef", "Node", "EinsumNode", "OpaqueNode", "VecGlueNode", "TensorOp",
    "Program", "StagedExecutor", "substrate_modes", "OpaqueRegistry",
    "OpaqueContract", "default_registry", "capture_mode", "GateResult",
    "frob_rel", "gate_frob_rel", "gate_outputs", "gate_determinism",
    "Planner", "LeverDecision", "format_decisions",
    "CaptureExecutor", "GraphReplayExecutor",
]
