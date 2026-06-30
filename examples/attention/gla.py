"""GLA (Gated Linear Attention) as a chunked pto-fuser forward.

GLA's decay is data-dependent and *per channel*: every d_k channel of the state
decays on its own schedule ``g_t = σ(·) ∈ (0,1)^{d_k}``. This is the most general gate
in the linear family — RetNet and Mamba-2 are the scalar special cases. See
:func:`attention.gate_gla`.

    python examples/attention/gla.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention import gate_gla, run_linear_variant  # noqa: E402

if __name__ == "__main__":
    run_linear_variant("GLA (gated linear attention)", gate_gla)
