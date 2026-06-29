"""Reference chunk-attention forwards built on the IR + executor."""
from .deltanet import build_deltanet_program, deltanet_reference, make_inputs
from .gdn import gdn_contraction_stages
from .fused_stages import (build_kkt_fused_program, build_scan_fused_program,
                           build_scan_staged_program, kkt_reference,
                           make_kkt_inputs, make_scan_inputs, scan_reference)

__all__ = ["build_deltanet_program", "deltanet_reference", "make_inputs",
           "gdn_contraction_stages",
           "make_scan_inputs", "scan_reference", "build_scan_staged_program",
           "build_scan_fused_program", "make_kkt_inputs", "kkt_reference",
           "build_kkt_fused_program"]
