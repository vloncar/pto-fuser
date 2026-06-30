"""RetNet (chunked retention) as a chunked pto-fuser forward.

Retention is linear attention with one fixed scalar decay γ per head (the standard
``γ_h = 1 − 2^(−5−h)`` schedule). In the shared chunked core that is just a constant,
per-head, position-independent gate — see :func:`attention.gate_retnet`.

    python examples/attention/retnet.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention import gate_retnet, run_linear_variant  # noqa: E402

if __name__ == "__main__":
    run_linear_variant("RetNet (retention)", gate_retnet)
