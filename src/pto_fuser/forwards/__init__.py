"""Reference chunk-attention forwards built on the IR + executor."""
from .deltanet import build_deltanet_program, deltanet_reference, make_inputs
from .gdn import gdn_contraction_stages

__all__ = ["build_deltanet_program", "deltanet_reference", "make_inputs",
           "gdn_contraction_stages"]
