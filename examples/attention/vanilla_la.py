"""Vanilla (non-gated) linear attention as a chunked pto-fuser forward.

The simplest member of the family: no decay (g_t = 1), so the state just accumulates
``S += kᵀv`` and ``o = qS``. Everything else — the chunk decomposition, the staged /
captured backends, the correctness gate — is shared with the gated variants in
``examples/attention/_chunked.py``.

    python examples/attention/vanilla_la.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention import gate_vanilla, run_linear_variant  # noqa: E402

if __name__ == "__main__":
    run_linear_variant("vanilla linear attention", gate_vanilla)
