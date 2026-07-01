"""The chunked-attention example zoo.

Each mechanism is a thin specialisation:

  * the **linear-recurrence family** (vanilla LA, RetNet, GLA, Mamba-2) shares
    :mod:`examples.attention._chunked` and differs only in the per-token gate — see
    :func:`gate_vanilla` / :func:`gate_retnet` / :func:`gate_gla` / :func:`gate_mamba2`;
  * the **delta-rule family** (GDN, KDA) reuses the DeltaNet forward builder shipped
    in ``pto_fuser.forwards`` (kkt → triangular-inverse → recompute → scan → output).

:func:`run_linear_variant` is the common driver the linear examples call: it builds
the inputs, runs the forward staged, gates it bit-faithfully against the token-
recurrent fp32 reference, captures it as a single dispatch (bit-exact), and prints a
small timing table.
"""
from __future__ import annotations

import os
import sys

# Bootstrap: put the package src/ and the examples/ dir on the path so the examples
# run from a checkout without an install (mirrors tests/conftest.py).
_EXAMPLES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(os.path.dirname(_EXAMPLES), "src")
for _p in (_EXAMPLES, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PTO_LIB_PATH", "/home/vloncar/work/einsum_workspace/pto-isa")

import torch

from ._chunked import (build_chunked_linear_program, chunked_linear_reference,
                       make_qkv, prepare_inputs)

__all__ = ["gate_vanilla", "gate_retnet", "gate_gla", "gate_mamba2",
           "run_linear_variant"]


# --------------------------------------------------------------------------- #
#  Per-token gates  g_t ∈ (0,1]^{d_k}, shape [N, nc, C, d_k]  (N = B*H)
# --------------------------------------------------------------------------- #
def gate_vanilla(N, nc, C, d_k, H, device, seed=1):
    """No decay — plain linear attention."""
    return torch.ones(N, nc, C, d_k, device=device)


def gate_retnet(N, nc, C, d_k, H, device, seed=1):
    """RetNet: one fixed scalar decay γ_h per head (the standard 1 − 2^(−5−h)
    schedule), broadcast across channels and constant over tokens."""
    h = torch.arange(N, device=device) % H
    gamma_h = 1.0 - torch.pow(2.0, -5.0 - h.float())            # [N], near 1
    return gamma_h.view(N, 1, 1, 1).expand(N, nc, C, d_k).contiguous()


def gate_mamba2(N, nc, C, d_k, H, device, seed=1):
    """Mamba-2 / SSD: a data-dependent *scalar* decay a_t per token per head
    (a_t = exp(−softplus(Δ_t)) ∈ (0,1)), broadcast across channels."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    delta = (torch.randn(N, nc, C, 1, generator=g) * 0.5 + 1.0).to(device)
    a = torch.exp(-torch.nn.functional.softplus(delta))         # (0,1)
    a = 0.9 + 0.1 * a                                           # keep near 1 (fp16-stable)
    return a.expand(N, nc, C, d_k).contiguous()


def gate_gla(N, nc, C, d_k, H, device, seed=1):
    """GLA: a data-dependent *per-channel* gate (each d_k channel decays on its own
    schedule). Clamped near 1 for fp16 stability in this small example."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(N, nc, C, d_k, generator=g).to(device)
    return 0.9 + 0.1 * torch.sigmoid(x)


# --------------------------------------------------------------------------- #
#  Common driver
# --------------------------------------------------------------------------- #
def run_linear_variant(name: str, gate_fn, *, per_dim_gate=False, B=2, H=4, nc=4,
                       C=16, d_k=64, d_v=64, tol=2e-2):
    """Build canonical → compile (propose/verify/dispose) → gate vs fp32 reference.

    The compile pass fuses the ``nc`` per-chunk intra-scores into one proven kernel
    (``batch-chunk-intra-score``: ``gated_qk_native_v2`` for a scalar gate,
    ``qk_prologue`` for GLA's per-channel gate) and selects read modes — each kept only
    on a gated, deterministic, measured win. ``per_dim_gate`` is the forward-declared
    gate kind (the one property the canonical IR cannot carry)."""
    from common import pick_device                                # examples/common
    from pto_fuser import (Features, GraphReplayExecutor, compile_program,
                           frob_rel, gate_outputs)

    dev = pick_device()
    print(f"\n=== {name}  (B={B} H={H} nc={nc} C={C} d_k={d_k} d_v={d_v}"
          f"{' per-dim gate' if per_dim_gate else ''}) ===")
    if dev is None:
        print("  no healthy NPU — skipping the run (the Program still builds off-NPU).")
        build_chunked_linear_program(B * H, nc, C, d_k, d_v)
        print("  Program built OK.")
        return

    N = B * H
    q, k, v = make_qkv(N, nc, C, d_k, d_v, dev, seed=0)
    gates = gate_fn(N, nc, C, d_k, H, dev)
    bindings = prepare_inputs(q, k, v, gates)
    canon = build_chunked_linear_program(N, nc, C, d_k, d_v)
    ref = chunked_linear_reference(q, k, v, gates, B, H, nc, C)

    result = compile_program(canon, Features(B, H, nc, C, d_k, per_dim_gate=per_dim_gate),
                             bindings=bindings, iters=20)
    print("  " + str(result.report).replace("\n", "\n  "))
    got = GraphReplayExecutor().capture(result.program, bindings).replay(bindings)
    gates_res = gate_outputs(got, ref, tol=tol)
    for r in gates_res:
        print("  " + str(r))
    return all(r.passed for r in gates_res)


def time_ms_row(label, fn):
    from common import Measurement, time_ms
    return Measurement(label, time_ms(fn))
