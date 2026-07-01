"""Fusion-region analysis (B0) unit tests (no NPU).

The analysis is shape-only (meta tensors), so it runs off-device on CPU bindings.
Covers the region partition, the opaque-boundary cut, the byte metrics, and the
safety property that grounds megakernel generation: **every structural transform we
apply stays inside a single identified region** (a fusion never spans the tri_inv
boundary).
"""
import torch

from pto_fuser import identify_fusion_regions


def _gdn():
    from attention._gdn_full import (build_gdn_program, make_gdn_inputs,
                                     prepare_gdn_bindings)
    B, H, nc, C, D = 1, 16, 8, 128, 128
    inp = make_gdn_inputs(B, H, nc, C, D, "cpu")
    binds = prepare_gdn_bindings(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g_in"])
    return build_gdn_program(B, H, nc, C, D, D ** -0.5), binds


def _kda():
    from attention._kda_full import (build_kda_program, make_kda_inputs,
                                     prepare_kda_bindings)
    B, H, nc, C, D = 1, 16, 8, 128, 128
    inp = make_kda_inputs(B, H, nc, C, D, "cpu")
    binds = prepare_kda_bindings(inp["q"], inp["k"], inp["v"], inp["beta"], inp["g_in"])
    return build_kda_program(B, H, nc, C, D, D ** -0.5), binds


# --------------------------------------------------------------------------- #
#  region partition + opaque boundary
# --------------------------------------------------------------------------- #
def test_gdn_two_regions_cut_at_tri_inv():
    prog, binds = _gdn()
    an = identify_fusion_regions(prog, binds)
    assert len(an.regions) == 2
    assert an.opaque == ["T_raw"]                     # the triangular inverse is the cut
    # the kkt cluster is a contraction-epilogue; the post-solve cluster is a recurrence
    kinds = {an.region_of("A").kind, an.region_of("S0").kind}
    assert kinds == {"contraction-epilogue", "recurrence"}


def test_kkt_region_is_the_gated_kkt_dropset():
    prog, binds = _gdn()
    an = identify_fusion_regions(prog, binds)
    r = an.region_of("Araw")
    assert r.kind == "contraction-epilogue"
    assert set(r.node_names) == {"Araw", "Ag", "A_t", "A"}   # exactly the kkt-gated template region
    assert r.inputs == sorted(["kF", "coefA"])               # boundary reads
    assert r.outputs == ["A"]                                # A feeds tri_inv (outside)
    assert set(r.internal) == {"Araw", "Ag", "A_t"}          # the qk round-trip fusion saves


def test_opaque_node_is_in_no_fusible_region():
    prog, binds = _gdn()
    an = identify_fusion_regions(prog, binds)
    assert an.region_of("T_raw") is None                     # opaque = hard boundary
    # A (before) and Tm (after) are on opposite sides of the cut
    assert an.region_of("A") is not an.region_of("Tm")


# --------------------------------------------------------------------------- #
#  the safety property: transforms stay inside one region
# --------------------------------------------------------------------------- #
def test_structural_transforms_are_region_contained():
    prog, binds = _gdn()
    an = identify_fusion_regions(prog, binds)
    # the kkt-gated template touches only the kkt cluster
    kkt = {an.region_of(n) for n in ("Araw", "Ag", "A_t", "A")}
    assert len(kkt) == 1
    # LowerResidentScan + the chunk-o template both live in the post-solve region
    scan = {an.region_of(n) for n in ("S0", "WS0", "dS0", "S1", "h_flat", "vn_flat")}
    chunk_o = {an.region_of(n) for n in ("Aqk", "Aqk_g", "Aqk_c", "o")}
    assert len(scan) == 1 and len(chunk_o) == 1
    assert scan == chunk_o                                    # same post-solve region
    assert next(iter(scan)) is not next(iter(kkt))           # different from the kkt region


def test_kda_regions_and_containment():
    prog, binds = _kda()
    an = identify_fusion_regions(prog, binds)
    assert len(an.regions) == 2 and an.opaque == ["T_raw"]
    assert an.region_of("Araw").kind == "contraction-epilogue"
    assert an.region_of("S0").kind == "recurrence"
    assert an.region_of("k_eff") is an.region_of("Aqk")      # KDA chunk_o glue in one region


# --------------------------------------------------------------------------- #
#  metrics + meta shape inference
# --------------------------------------------------------------------------- #
def test_internal_bytes_quantify_the_opportunity():
    prog, binds = _gdn()
    an = identify_fusion_regions(prog, binds)
    # the recurrence region dominates the on-chip opportunity
    rec = an.region_of("S0")
    assert rec.internal_bytes > an.region_of("A").internal_bytes
    assert an.total_internal_bytes == sum(r.internal_bytes for r in an.regions)
    # ranked() is descending by score
    scores = [r.score for r in an.ranked()]
    assert scores == sorted(scores, reverse=True)


def test_meta_shapes_match_real_dims():
    # Araw = kF·kFᵀ over "nid,njd->nij" at M=B*H*nc, C=128 -> [M,C,C] fp16
    from pto_fuser.analysis import _infer_shapes
    prog, binds = _gdn()
    shapes = _infer_shapes(prog, binds)
    M = 1 * 16 * 8
    assert tuple(shapes["Araw"].shape) == (M, 128, 128)
    assert shapes["Araw"].dtype == torch.float16
    assert tuple(shapes["A"].shape) == (M, 128, 128)
