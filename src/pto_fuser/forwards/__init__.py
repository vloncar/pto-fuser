"""Reference chunk-attention forwards built on the IR + executor."""
from .deltanet import build_deltanet_program, deltanet_reference, make_inputs

__all__ = ["build_deltanet_program", "deltanet_reference", "make_inputs"]
