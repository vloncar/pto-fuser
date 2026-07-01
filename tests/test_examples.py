"""Examples smoke tests (no NPU) — every example imports and builds its Program.

These keep the examples honest as the package evolves: an API rename that breaks an
example is caught here, off-NPU, without needing a device (the on-device runs are the
NPU forward/fusion tests). Only the import + Program-construction path is exercised.
"""
import importlib

import pytest

EXAMPLE_MODULES = [
    "common", "minimal",
    "attention", "attention._chunked", "attention._gdn_full", "attention._kda_full",
    "attention.vanilla_la", "attention.retnet", "attention.gla", "attention.mamba2",
    "attention.gdn", "attention.kda",
    "workflow.read_modes", "workflow.graph_capture", "workflow.fusion_decision",
    "benchmarks.gdn_features", "benchmarks._mega_bench",
    "benchmarks.gdn_mega", "benchmarks.kda_mega",
]


@pytest.mark.parametrize("mod", EXAMPLE_MODULES)
def test_example_imports(mod):
    importlib.import_module(mod)


def test_minimal_and_chunked_programs_build():
    import minimal
    from attention._chunked import build_chunked_linear_program
    assert minimal.build_program(0.125).outputs == ["attn"]
    assert build_chunked_linear_program(8, 4, 16, 64, 64).outputs == ["o"]


def test_chunked_fused_intra_program_builds():
    """fused_intra (scalar-gated zoo) replaces the per-chunk A einsum + tril with ONE
    gated_qk_native_v2 FusedNode over all M=N·nc chunks; the staged default keeps the
    einsum path. Verifies the opt-in lever's structure + extra scalar-gate bindings."""
    from pto_fuser import EinsumNode, FusedNode
    from attention._chunked import build_chunked_linear_program, prepare_inputs, make_qkv
    from attention import gate_retnet
    N, nc, C, d_k, d_v = 8, 4, 16, 64, 64
    prog = build_chunked_linear_program(N, nc, C, d_k, d_v, fused_intra=True)
    # one fused gated score over all chunks, no per-chunk q̃k̂ᵀ einsum (eq "nid,njd->nij")
    assert any(isinstance(n, FusedNode) and n.kernel == "gated_qk_native_v2" for n in prog.nodes)
    assert not any(isinstance(n, EinsumNode) and n.equation == "nid,njd->nij" for n in prog.nodes)
    assert prog.inputs[-2:] == ["g_intra", "beta_intra"]
    # staged default still carries the per-chunk score einsum and no FusedNode
    staged = build_chunked_linear_program(N, nc, C, d_k, d_v)
    assert any(isinstance(n, EinsumNode) and n.equation == "nid,njd->nij" for n in staged.nodes)
    assert not any(isinstance(n, FusedNode) for n in staged.nodes)
    # the scalar-gate bindings prepare_inputs adds match the new Program inputs
    q, k, v = make_qkv(N, nc, C, d_k, d_v, "cpu")
    binds = prepare_inputs(q, k, v, gate_retnet(N, nc, C, d_k, 4, "cpu"))
    assert binds["g_intra"].shape == (N * nc, C) and binds["beta_intra"].shape == (N * nc, C)


def test_chunked_fused_intra_per_dim_program_builds():
    """per_dim_gate (GLA/KDA): the intra score lowers to the qk_prologue FusedNode
    (operand prescale q⊙P/k⊙invP ahead of the matmul) over the existing P/invP decay
    tensors — no g_intra/beta_intra (those are the scalar-epilogue path's)."""
    from pto_fuser import EinsumNode, FusedNode
    from attention._chunked import build_chunked_linear_program
    N, nc, C, d_k, d_v = 8, 4, 16, 64, 64
    prog = build_chunked_linear_program(N, nc, C, d_k, d_v, fused_intra=True, per_dim_gate=True)
    assert any(isinstance(n, FusedNode) and n.kernel == "qk_prologue" for n in prog.nodes)
    assert not any(isinstance(n, EinsumNode) and n.equation == "nid,njd->nij" for n in prog.nodes)
    # per-dim uses P/invP (already inputs); it does NOT add the scalar-epilogue bindings
    assert "g_intra" not in prog.inputs and "beta_intra" not in prog.inputs
    pro = [n for n in prog.nodes if isinstance(n, FusedNode)][0]
    assert pro.inputs == ["q", "k", "P", "invP"]
    # prologue_v2 selects the L2-resident ring kernel (same inputs/structure)
    progv2 = build_chunked_linear_program(N, nc, C, d_k, d_v, fused_intra=True,
                                          per_dim_gate=True, prologue_v2=True)
    assert any(isinstance(n, FusedNode) and n.kernel == "qk_prologue_v2" for n in progv2.nodes)


def test_full_gdn_kda_programs_build():
    """The end-to-end gated forwards build + their host-coefficient bindings match the
    Program inputs (off-NPU; the on-device numeric gate is benchmarks/{gdn,kda}_mega)."""
    from attention._gdn_full import (build_gdn_program, make_gdn_inputs,
                                     prepare_gdn_bindings)
    from attention._kda_full import (build_kda_program, make_kda_inputs,
                                     prepare_kda_bindings)
    B, H, nc, C, D = 1, 4, 2, 128, 128
    for make, build, prep in [(make_gdn_inputs, build_gdn_program, prepare_gdn_bindings),
                              (make_kda_inputs, build_kda_program, prepare_kda_bindings)]:
        inp = make(B, H, nc, C, D, "cpu")
        prog = build(B, H, nc, C, D, D ** -0.5)
        binds = prep(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g_in"])
        assert prog.outputs == ["o"]
        assert sorted(prog.inputs) == sorted(binds.keys())


def test_gdn_fused_scan_program_builds():
    """The chunk_h_scan resident-state lowering builds, hosts the FusedNode, and keeps
    the same inputs/output as the einsum scan (the on-device gate is benchmarks/gdn_mega)."""
    from pto_fuser import FusedNode
    from attention._gdn_full import (build_gdn_program, make_gdn_inputs,
                                     prepare_gdn_bindings)
    B, H, nc, C, D = 1, 4, 2, 128, 128
    inp = make_gdn_inputs(B, H, nc, C, D, "cpu")
    binds = prepare_gdn_bindings(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g_in"])
    prog = build_gdn_program(B, H, nc, C, D, D ** -0.5, fused_scan=True)
    assert prog.outputs == ["o"]
    assert sorted(prog.inputs) == sorted(binds.keys())
    assert any(isinstance(n, FusedNode) and n.kernel == "chunk_h_scan" for n in prog.nodes)


def test_gdn_fused_kkt_program_builds():
    """The kkt_gated lowering builds, hosts the FusedNode, and keeps the same
    inputs/output as the einsum kkt. Default fused_native=True selects the native
    [M,C,D] kernel (no layout bridge); fused_native=False the mega-bridge kernel."""
    from pto_fuser import FusedNode, TensorOp
    from attention._gdn_full import (build_gdn_program, make_gdn_inputs,
                                     prepare_gdn_bindings)
    B, H, nc, C, D = 1, 4, 2, 128, 128
    inp = make_gdn_inputs(B, H, nc, C, D, "cpu")
    binds = prepare_gdn_bindings(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g_in"])
    # native (default): no transpose bridge, A produced directly
    prog = build_gdn_program(B, H, nc, C, D, D ** -0.5, fused_scan=True, fused_kkt=True)
    assert prog.outputs == ["o"]
    assert sorted(prog.inputs) == sorted(binds.keys())
    assert any(isinstance(n, FusedNode) and n.kernel == "kkt_gated_native" for n in prog.nodes)
    assert not any(isinstance(n, TensorOp) and n.output == "k_kkt" for n in prog.nodes)
    # mega bridge
    megap = build_gdn_program(B, H, nc, C, D, D ** -0.5, fused_scan=True, fused_kkt=True,
                              fused_native=False)
    assert any(isinstance(n, FusedNode) and n.kernel == "kkt_gated" for n in megap.nodes)


def test_gdn_fused_chunk_o_program_builds():
    """chunk_o's Aqk lowers to the shared gated_qk FusedNode (q@kᵀ + on-chip
    gate/causal-mask epilogue); same inputs/output as the einsum path. Native default."""
    from pto_fuser import FusedNode
    from attention._gdn_full import (build_gdn_program, make_gdn_inputs,
                                     prepare_gdn_bindings)
    B, H, nc, C, D = 1, 4, 2, 128, 128
    inp = make_gdn_inputs(B, H, nc, C, D, "cpu")
    binds = prepare_gdn_bindings(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g_in"])
    prog = build_gdn_program(B, H, nc, C, D, D ** -0.5, fused_scan=True, fused_chunk_o=True)
    assert prog.outputs == ["o"]
    assert sorted(prog.inputs) == sorted(binds.keys())
    assert any(isinstance(n, FusedNode) and n.kernel == "gated_qk_native" for n in prog.nodes)
    megap = build_gdn_program(B, H, nc, C, D, D ** -0.5, fused_scan=True, fused_chunk_o=True,
                              fused_native=False)
    assert any(isinstance(n, FusedNode) and n.kernel == "gated_qk" for n in megap.nodes)


def test_kda_fused_scan_program_builds():
    """KDA's chunk_h_scan lowering builds with the per-dim-decay FusedNode and keeps
    the same inputs/output as the einsum scan (on-device gate is benchmarks/kda_mega)."""
    from pto_fuser import FusedNode
    from attention._kda_full import (build_kda_program, make_kda_inputs,
                                     prepare_kda_bindings)
    B, H, nc, C, D = 1, 4, 2, 128, 128
    inp = make_kda_inputs(B, H, nc, C, D, "cpu")
    binds = prepare_kda_bindings(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g_in"])
    prog = build_kda_program(B, H, nc, C, D, D ** -0.5, fused_scan=True)
    assert prog.outputs == ["o"]
    assert sorted(prog.inputs) == sorted(binds.keys())
    fn = [n for n in prog.nodes if isinstance(n, FusedNode) and n.kernel == "chunk_h_scan"]
    assert len(fn) == 1 and fn[0].params.get("perdim_decay") is True
