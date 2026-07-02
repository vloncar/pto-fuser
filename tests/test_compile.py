"""Cost model / policy / compile-driver unit tests (no NPU).

The propose/verify/dispose loop with verification off: exercises feature derivation,
the seeded cost predictions, the policy's ordered pipeline, and that
``compile_program(verify=False)`` lowers the canonical program to the expected fused
form with a coherent report.
"""
from pto_fuser import (CostModel, Features, Policy, compile_program, canonicalize,
                       EinsumNode, FusedNode)


def _gdn():
    from attention._gdn_full import build_gdn_program
    return build_gdn_program(1, 16, 8, 128, 128, 128 ** -0.5)


def _kda():
    from attention._kda_full import build_kda_program
    return build_kda_program(1, 16, 8, 128, 128, 128 ** -0.5)


# --------------------------------------------------------------------------- #
#  features + cost model
# --------------------------------------------------------------------------- #
def test_features_derived():
    f = Features(1, 16, 8, 128, 128)
    assert f.T == 8 * 128 and f.M == 1 * 16 * 8 and f.N == 1 * 16


def test_regime_classification():
    assert Features(1, 1, 8, 128, 128).regime == "launch-bound"
    assert Features(1, 8, 8, 128, 128).regime == "crossover"
    assert Features(1, 16, 8, 128, 128).regime == "bandwidth-bound"


def test_cost_predictions_worth_trying():
    cm, f = CostModel(), Features(1, 16, 8, 128, 128)
    for name in ("enable-direct-reads", "lower-resident-scan",
                 "fuse-contraction-epilogue"):
        assert cm.predict(name, f).worth_trying
    assert not cm.predict("unknown", f).worth_trying


def test_cost_v2_shape_gate():
    cm = CostModel()
    assert cm.predict("fuse-contraction-epilogue", Features(1, 4, 8, 128, 128)).v2 is True
    assert cm.predict("fuse-contraction-epilogue", Features(1, 4, 8, 16, 64)).v2 is False


# --------------------------------------------------------------------------- #
#  policy pipeline
# --------------------------------------------------------------------------- #
def test_policy_orders_fusions_before_reads():
    canon = canonicalize(_gdn())
    plan = Policy().pipeline(canon, Features(1, 16, 8, 128, 128))
    names = [p.name for p in plan]
    # only GDN-applicable transforms proposed (no KDA per-dim scan)
    assert "lower-perdim-scan" not in names
    # structural fusions precede the annotation levers
    assert names.index("lower-resident-scan") < names.index("enable-direct-reads")
    assert names.index("fuse-contraction-epilogue") < names.index("enable-direct-reads")


def test_policy_prunes_by_match():
    canon = canonicalize(_kda())
    plan = Policy().pipeline(canon, Features(1, 16, 8, 128, 128))
    names = [p.name for p in plan]
    # KDA: per-dim scan + the epilogue generator (its qk_prologue template) apply;
    # the GDN scalar-decay scan does not.
    assert "lower-perdim-scan" in names and "fuse-contraction-epilogue" in names
    assert "lower-resident-scan" not in names


# --------------------------------------------------------------------------- #
#  compile driver (unverified)
# --------------------------------------------------------------------------- #
def test_compile_gdn_unverified():
    res = compile_program(_gdn(), Features(1, 16, 8, 128, 128), verify=False)
    kernels = {n.kernel for n in res.program.nodes if isinstance(n, FusedNode)}
    # chunk_o fuses score→output (qkv_flash_native_v2, the double-buffered interleave);
    # the epilogue generator then emits only kkt (v2 gate), score already consumed.
    assert kernels == {"chunk_h_scan", "kkt_gated_native_v2", "qkv_flash_native_v2"}
    # every surviving einsum promoted to a direct read
    assert all(n.read_mode == "auto" for n in res.program.nodes
               if isinstance(n, EinsumNode))
    assert not res.report.verified
    # resident-scan + chunk-o-flash + contraction-epilogue + 2 read levers = 5 transforms
    assert len(res.report.kept) == len(res.report.records) == 5


def test_compile_kda_unverified():
    res = compile_program(_kda(), Features(1, 16, 8, 128, 128), verify=False)
    kernels = {n.kernel for n in res.program.nodes if isinstance(n, FusedNode)}
    # per-dim chunk_o flashes (consuming the score the prologue template would have fused);
    # scan + perdim-flash + 2 read levers = 4 kept.
    assert kernels == {"chunk_h_scan", "qkvp_flash_native_v2"}
    assert len(res.report.kept) == 4


def test_report_str_is_readable():
    res = compile_program(_gdn(), Features(1, 16, 8, 128, 128), verify=False)
    text = str(res.report)
    assert "transforms kept" in text and "lower-resident-scan" in text
