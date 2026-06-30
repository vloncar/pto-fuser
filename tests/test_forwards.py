"""Forward-builder tests — structure + chunked-decomposition correctness (no NPU).

The chunked linear-attention forward (shared by the vanilla LA / RetNet / GLA /
Mamba-2 examples) is validated *off-NPU*: its eager-torch mirror must reproduce the
token-recurrent reference for every gate type, which proves the chunk decomposition
itself is correct independently of the device kernels. The Program structure is
checked here too so the builder can't silently drift.
"""
import pytest
import torch

from attention import gate_gla, gate_mamba2, gate_retnet, gate_vanilla
from attention._chunked import (build_chunked_linear_program, chunked_linear_reference,
                                chunked_linear_torch, make_qkv)

VARIANTS = {"vanilla": gate_vanilla, "retnet": gate_retnet,
            "gla": gate_gla, "mamba2": gate_mamba2}


def test_chunked_program_structure():
    B, H, nc, C, d_k, d_v = 2, 4, 4, 16, 64, 64
    prog = build_chunked_linear_program(B * H, nc, C, d_k, d_v)
    assert prog.inputs == ["q", "k", "v", "P", "invP", "gammaInvP", "gamma"]
    assert prog.outputs == ["o"]
    from pto_fuser import EinsumNode
    # 4 einsum cores per chunk (intra qk, intra·v, inter, state-update).
    assert sum(isinstance(n, EinsumNode) for n in prog.nodes) == 4 * nc


@pytest.mark.parametrize("name", list(VARIANTS))
def test_chunk_decomposition_matches_recurrent(name):
    """The chunked formula (what the Program encodes) must equal the token-recurrent
    definition for each variant's gate, on CPU, to fp32 roundoff."""
    B, H, nc, C, d_k, d_v = 2, 4, 4, 16, 32, 32
    N = B * H
    q, k, v = make_qkv(N, nc, C, d_k, d_v, "cpu", seed=0)
    gates = VARIANTS[name](N, nc, C, d_k, H, "cpu")
    chunked = chunked_linear_torch(q, k, v, gates, B, H, nc, C)["o"]
    recurrent = chunked_linear_reference(q, k, v, gates, B, H, nc, C)["o"]
    rel = (chunked - recurrent).norm() / recurrent.norm().clamp_min(1e-9)
    assert rel < 1e-4, f"{name}: chunked vs recurrent rel={rel:.2e}"


def test_gate_shapes_and_range():
    """Every gate generator returns [N, nc, C, d_k] in (0, 1]."""
    N, nc, C, d_k, H = 8, 3, 16, 32, 4
    for name, fn in VARIANTS.items():
        g = fn(N, nc, C, d_k, H, "cpu")
        assert g.shape == (N, nc, C, d_k), name
        assert g.min() > 0 and g.max() <= 1.0 + 1e-6, name
