"""Template-emission (B1) unit tests (no NPU).

Covers epilogue-unit extraction, template matching/emission, the region-driven
generator, and the proven-only guarantee (a contraction whose epilogue matches no
template is left staged — the fixpipe-1D finding).
"""
from pto_fuser import (EinsumNode, FusedNode, canonicalize, epilogue_report,
                       extract_epilogue_unit, FuseContractionEpilogue)
from pto_fuser.template import (GatedKKTTemplate, GatedChunkOTemplate,
                                PerDimPrologueTemplate, default_templates)


def _gdn():
    from attention._gdn_full import build_gdn_program
    return canonicalize(build_gdn_program(1, 4, 2, 128, 128, 128 ** -0.5))


def _kda():
    from attention._kda_full import build_kda_program
    return canonicalize(build_kda_program(1, 4, 2, 128, 128, 128 ** -0.5))


def _einsum_idx(program, out):
    for i, n in enumerate(program.nodes):
        if isinstance(n, EinsumNode) and n.output == out:
            return i
    raise KeyError(out)


# --------------------------------------------------------------------------- #
#  extraction
# --------------------------------------------------------------------------- #
def test_extract_kkt_epilogue_unit():
    prog = _gdn()
    unit = extract_epilogue_unit(prog, _einsum_idx(prog, "Araw"))
    assert unit.epilogue_ops == ["mul", "tril", "contiguous"]
    assert unit.boundary_out == "A"                    # the tensor tri_inv consumes
    assert unit.anchor.inputs == ["kF", "kF"]


def test_extract_kda_prologue():
    prog = _kda()
    unit = extract_epilogue_unit(prog, _einsum_idx(prog, "Aqk"))
    # q_eff / k_eff prescales are the prologue; tril+contiguous the epilogue
    assert unit.epilogue_ops == ["tril", "contiguous"]
    assert set(unit.prologue) == {"q_eff", "k_eff"}


# --------------------------------------------------------------------------- #
#  template matching + emission
# --------------------------------------------------------------------------- #
def test_templates_match_their_own_pattern():
    gdn, kda = _gdn(), _kda()
    kkt = extract_epilogue_unit(gdn, _einsum_idx(gdn, "Araw"))
    aqk = extract_epilogue_unit(gdn, _einsum_idx(gdn, "Aqk"))
    pro = extract_epilogue_unit(kda, _einsum_idx(kda, "Aqk"))
    assert GatedKKTTemplate().matches(gdn, kkt)
    assert not GatedKKTTemplate().matches(gdn, aqk)
    assert GatedChunkOTemplate().matches(gdn, aqk)
    assert not GatedChunkOTemplate().matches(gdn, kkt)
    assert PerDimPrologueTemplate().matches(kda, pro)
    assert not PerDimPrologueTemplate().matches(gdn, kkt)


def test_generator_emits_expected_nodes():
    gdn = FuseContractionEpilogue(1, 4, 2, 128, 128, v2=False).apply(_gdn()).program
    fused = {n.kernel: n for n in gdn.nodes if isinstance(n, FusedNode)}
    assert set(fused) == {"kkt_gated_native", "gated_qk_native"}
    assert fused["kkt_gated_native"].inputs == ["kF", "g_native", "beta_native"]
    assert fused["gated_qk_native"].params.get("causal") is True


# --------------------------------------------------------------------------- #
#  proven-only guarantee + report
# --------------------------------------------------------------------------- #
def test_report_matches_two_and_flags_the_rest():
    rep = epilogue_report(_gdn())
    verdicts = {anchor: verdict for anchor, _, verdict in rep.rows}
    assert verdicts["Araw"] == "template kkt-gated-native"
    assert verdicts["Aqk"] == "template chunk-o-gated-native"
    # the per-row coef_qg scale on o_inter is NOT a proven template (fixpipe-1D finding)
    assert verdicts["o_inter_m"] == "staged (no matching template)"
    assert sum(v.startswith("template ") for v in verdicts.values()) == 2


def test_generator_never_emits_without_a_template():
    # o_inter_m (einsum + per-row mul) has no matching template -> left staged
    prog = FuseContractionEpilogue(1, 4, 2, 128, 128).apply(_gdn()).program
    assert any(isinstance(n, EinsumNode) and n.output == "o_inter_m" for n in prog.nodes)
    assert not any(isinstance(n, FusedNode) and n.output == "o_inter" for n in prog.nodes)


def test_default_templates_are_registered():
    names = {t.name for t in default_templates()}
    assert names == {"kkt-gated-native", "chunk-o-gated-native", "qk-prologue"}


def test_chunk_o_flash_transform():
    """FuseChunkOFlash fuses GDN's chunk_o score→output (Aqk einsum + coef_o mul +
    contiguous + o_intra A@v) into ONE qkv_flash_native node, distinct from the epilogue
    generator; it matches only that scalar-gate chunk_o chain (not KDA's per-dim one)."""
    from pto_fuser import EinsumNode, FusedNode, FuseChunkOFlash
    gdn = _gdn()
    t = FuseChunkOFlash(1, 4, 2, 128, 128)
    assert t.match(gdn) == 1
    assert t.match(_kda()) == 0                     # per-dim chunk_o -> no scalar match
    prog = t.apply(gdn).program
    flash = [n for n in prog.nodes if isinstance(n, FusedNode)
             and n.kernel == "qkv_flash_native_v2"]
    assert len(flash) == 1
    assert flash[0].outputs == ["o_intra"] and flash[0].params.get("causal") is True
    assert flash[0].inputs == ["qF", "kF", "vn_flat", "g_native", "beta_native_ones"]
    # the score matmul + its glue + the o_intra matmul are all gone
    assert not any(isinstance(n, EinsumNode) and n.output in ("Aqk", "o_intra")
                   for n in prog.nodes)
    assert t.match(prog) == 0                        # idempotent
