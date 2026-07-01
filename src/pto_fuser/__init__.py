"""pto-fuser — the fusion / auto-fuser layer over the pto-einsum library.

Surface: the three-node IR, the execution backends (staged / graph-replay / fused-node),
the opaque-kernel registry, the correctness gates, and the separated compilation stack —
transforms (`transform.py`, `transforms/`), cost model (`cost.py`), policy (`policy.py`),
and the propose/verify/dispose driver (`compile.py`). See docs/DESIGN.md.
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
from .transform import (Transform, TransformResult, canonicalize,
                        EnableDirectReads, EnableFusedStore)
from .cost import CostModel, Features, Prediction
from .policy import Policy, PlannedTransform
from .report import CompilationReport, TransformRecord
from .compile import CompileResult, compile_program

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
    # transform / heuristic / verification separation
    "Transform", "TransformResult", "canonicalize",
    "EnableDirectReads", "EnableFusedStore",
    "CostModel", "Features", "Prediction",
    "Policy", "PlannedTransform",
    "CompilationReport", "TransformRecord",
    "CompileResult", "compile_program",
]
