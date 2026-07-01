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


def test_select_prologue_kernel_shape_gate():
    """The per-dim prologue picks V1 at tiny tiles and V2 where the prescale is
    bandwidth-heavy (C·d_k >= 4096, the measured win regime); build_*_program
    auto-selects when prologue_v2 is left None (the default)."""
    from pto_fuser import FusedNode
    from attention._chunked import select_prologue_kernel, build_chunked_linear_program
    assert select_prologue_kernel(16, 64) == "qk_prologue"       # 1024, tiny tile
    assert select_prologue_kernel(64, 128) == "qk_prologue_v2"   # 8192, win regime
    # auto-select (prologue_v2=None) flows the shape gate into the Program
    big = build_chunked_linear_program(8, 4, 64, 128, 128, fused_intra=True, per_dim_gate=True)
    assert any(isinstance(n, FusedNode) and n.kernel == "qk_prologue_v2" for n in big.nodes)


def test_kda_qk_prologue_transform():
    """The AbsorbQKPrologue transform rewrites KDA chunk_o's intra score into the
    per-dim qk_prologue FusedNode (the per-channel analog of GDN's scalar gated_qk
    epilogue) over qF/kF and the exp(±g) coefficients; same output. Shape gate picks
    V2 at C=D=128; v2=False forces V1; the canonical program has no prologue node."""
    from pto_fuser import FusedNode
    from pto_fuser.transforms import AbsorbQKPrologue
    from attention._kda_full import build_kda_program
    B, H, nc, C, D = 1, 4, 2, 128, 128
    canon = build_kda_program(B, H, nc, C, D, D ** -0.5)
    assert not any(isinstance(n, FusedNode) and n.kernel.startswith("qk_prologue")
                   for n in canon.nodes)                        # canonical: no prologue
    prog = AbsorbQKPrologue(B, H, nc, C, D).apply(canon).program
    assert prog.outputs == ["o"]
    pro = [n for n in prog.nodes if isinstance(n, FusedNode) and n.kernel.startswith("qk_prologue")]
    assert len(pro) == 1 and pro[0].outputs == ["Aqk_c"]
    assert pro[0].inputs == ["qF", "kF", "coef_ag", "coef_bg"]
    assert pro[0].kernel == "qk_prologue_v2"                    # C·D = 16384 -> win regime
    v1 = AbsorbQKPrologue(B, H, nc, C, D, v2=False).apply(canon).program
    assert any(isinstance(n, FusedNode) and n.kernel == "qk_prologue" for n in v1.nodes)


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


def test_gdn_resident_scan_transform():
    """LowerResidentScan rewrites GDN's unrolled scan into the chunk_h_scan FusedNode
    and keeps the same output; the canonical program has no fused node and still
    validates its bindings (on-device gate is benchmarks/gdn_mega)."""
    from pto_fuser import FusedNode
    from pto_fuser.transforms import LowerResidentScan
    from attention._gdn_full import (build_gdn_program, make_gdn_inputs,
                                     prepare_gdn_bindings)
    B, H, nc, C, D = 1, 4, 2, 128, 128
    inp = make_gdn_inputs(B, H, nc, C, D, "cpu")
    binds = prepare_gdn_bindings(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g_in"])
    canon = build_gdn_program(B, H, nc, C, D, D ** -0.5)
    assert sorted(canon.inputs) == sorted(binds.keys())
    assert not any(isinstance(n, FusedNode) for n in canon.nodes)     # canonical: staged
    prog = LowerResidentScan(B, H, nc, C, D).apply(canon).program
    assert prog.outputs == ["o"]
    fn = [n for n in prog.nodes if isinstance(n, FusedNode) and n.kernel == "chunk_h_scan"]
    assert len(fn) == 1 and "perdim_decay" not in fn[0].params      # GDN = scalar decay


def test_gdn_gated_kkt_transform():
    """AbsorbGatedKKT rewrites the kkt einsum+coefA mul+tril into kkt_gated_native
    (native [M,C,D], no transpose bridge); v2 selects the FFTS-interleave kernel."""
    from pto_fuser import FusedNode, TensorOp
    from pto_fuser.transforms import AbsorbGatedKKT
    from attention._gdn_full import build_gdn_program
    B, H, nc, C, D = 1, 4, 2, 128, 128
    canon = build_gdn_program(B, H, nc, C, D, D ** -0.5)
    prog = AbsorbGatedKKT(nc, H).apply(canon).program
    kkt = [n for n in prog.nodes if isinstance(n, FusedNode) and n.kernel.startswith("kkt_gated_native")]
    assert len(kkt) == 1 and kkt[0].outputs == ["A"]
    assert kkt[0].inputs == ["kF", "g_native", "beta_native"]
    assert kkt[0].kernel == "kkt_gated_native"                        # v1 default
    assert not any(isinstance(n, TensorOp) and n.output == "k_kkt" for n in prog.nodes)
    v2 = AbsorbGatedKKT(nc, H, v2=True).apply(canon).program
    assert any(isinstance(n, FusedNode) and n.kernel == "kkt_gated_native_v2" for n in v2.nodes)


def test_gdn_gated_chunk_o_transform():
    """AbsorbGatedChunkO rewrites chunk_o's Aqk einsum+coef_o mul+contiguous into the
    native gated_qk FusedNode (q@kᵀ + on-chip gate/causal-mask epilogue, β=1)."""
    from pto_fuser import FusedNode
    from pto_fuser.transforms import AbsorbGatedChunkO
    from attention._gdn_full import build_gdn_program
    B, H, nc, C, D = 1, 4, 2, 128, 128
    canon = build_gdn_program(B, H, nc, C, D, D ** -0.5)
    prog = AbsorbGatedChunkO(nc, H).apply(canon).program
    qk = [n for n in prog.nodes if isinstance(n, FusedNode) and n.kernel.startswith("gated_qk_native")]
    assert len(qk) == 1 and qk[0].outputs == ["Aqk_c"]
    assert qk[0].inputs == ["qF", "kF", "g_native", "beta_native_ones"]
    assert qk[0].params.get("causal") is True


def test_kda_perdim_scan_transform():
    """LowerPerDimScan rewrites KDA's unrolled scan into the per-dim-decay chunk_h_scan
    FusedNode; the canonical program still validates its bindings."""
    from pto_fuser import FusedNode
    from pto_fuser.transforms import LowerPerDimScan
    from attention._kda_full import (build_kda_program, make_kda_inputs,
                                     prepare_kda_bindings)
    B, H, nc, C, D = 1, 4, 2, 128, 128
    inp = make_kda_inputs(B, H, nc, C, D, "cpu")
    binds = prepare_kda_bindings(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g_in"])
    canon = build_kda_program(B, H, nc, C, D, D ** -0.5)
    assert sorted(canon.inputs) == sorted(binds.keys())
    prog = LowerPerDimScan(B, H, nc, C, D).apply(canon).program
    assert prog.outputs == ["o"]
    fn = [n for n in prog.nodes if isinstance(n, FusedNode) and n.kernel == "chunk_h_scan"]
    assert len(fn) == 1 and fn[0].params.get("perdim_decay") is True
