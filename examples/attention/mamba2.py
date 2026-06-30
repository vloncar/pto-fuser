"""Mamba-2 (State-Space Duality / SSD) as a chunked pto-fuser forward.

SSD's chunked scan is, in linear-attention terms, a data-dependent *scalar* decay
``a_t = exp(−softplus(Δ_t)) ∈ (0,1)`` per token per head (broadcast over channels),
with B_t playing the role of the key and C_t the role of the query. Structurally it
is the linear-recurrence core with a scalar gate — distinct from the delta-rule
family (no triangular inverse), which is why it is a good second topology to show.
See :func:`attention.gate_mamba2`.

    python examples/attention/mamba2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention import gate_mamba2, run_linear_variant  # noqa: E402

if __name__ == "__main__":
    run_linear_variant("Mamba-2 (SSD)", gate_mamba2)
