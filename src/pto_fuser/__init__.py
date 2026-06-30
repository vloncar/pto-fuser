"""pto-fuser — the fusion / auto-fuser layer over the pto-einsum library.

Surface: the three-node IR, the staged executor, the opaque-kernel registry, the
correctness gates, the read-mode / fused-store Planner, the graph-replay backend
(dispatch elimination), and the fused-node backend + the staged-vs-fused decision. See docs/FUSER_DESIGN.md (design) and docs/IMPLEMENTATION.md
(realized state).
"""
from .ir import (EinsumNode, FusedNode, Node, OpaqueNode, Program, TensorOp,
                 TensorRef, VecGlueNode)
from .executor import StagedExecutor, library_modes
from .registry import (OpaqueContract, OpaqueRegistry, capture_mode,
                       default_registry)
from .fused import (FusedContract, FusedKernel, FusedKernelRegistry,
                    default_fused_registry, shared_fused_registry)
from .gate import (GateResult, frob_rel, gate_determinism, gate_frob_rel,
                   gate_outputs)
from .planner import LeverDecision, Planner, format_decisions
from .graph import CaptureExecutor, GraphReplayExecutor
from .fusion import FusionDecision, decide, format_decisions as format_fusion_decisions

__all__ = [
    "TensorRef", "Node", "EinsumNode", "OpaqueNode", "VecGlueNode", "TensorOp",
    "FusedNode", "Program", "StagedExecutor", "library_modes", "OpaqueRegistry",
    "OpaqueContract", "default_registry", "capture_mode", "GateResult",
    "FusedContract", "FusedKernel", "FusedKernelRegistry",
    "default_fused_registry", "shared_fused_registry",
    "frob_rel", "gate_frob_rel", "gate_outputs", "gate_determinism",
    "Planner", "LeverDecision", "format_decisions",
    "CaptureExecutor", "GraphReplayExecutor",
    "FusionDecision", "decide", "format_fusion_decisions",
]
