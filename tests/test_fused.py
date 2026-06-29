"""M4 structural tests (no NPU): the FusedNode IR, multi-output plumbing, the
fused-kernel registry surface, and the FusionDecision verdict logic."""
import pytest

from pto_fuser import (FusedNode, Program, default_fused_registry, FusionDecision,
                       format_fusion_decisions)
from pto_fuser.ir import node_outputs
from pto_fuser.forwards import (build_scan_fused_program, build_scan_staged_program,
                                build_kkt_fused_program)


def test_fused_node_is_multi_output():
    n = FusedNode(kernel="chunk_h_scan", inputs=["w", "u", "k", "decay"],
                  outputs=["h_out", "final"], params={"B": 1, "H": 2, "nc": 3})
    assert node_outputs(n) == ["h_out", "final"]
    assert n.output == "h_out"            # primary, for messages


def test_program_accepts_multi_output_fused_node():
    # A FusedNode's several outputs must satisfy later use-before-def + declared
    # outputs (Program.__post_init__ adds every produced name).
    prog = build_scan_fused_program(2, 4, 3)
    assert prog.outputs == ["h_out", "final"]
    assert len(prog.nodes) == 1 and isinstance(prog.nodes[0], FusedNode)


def test_program_rejects_undeclared_fused_output():
    bad = FusedNode(kernel="k", inputs=["x"], outputs=["a", "b"], params={})
    with pytest.raises(ValueError):
        Program(nodes=[bad], inputs=["x"], outputs=["a", "c"])   # 'c' never produced


def test_staged_and_fused_scan_share_io():
    staged = build_scan_staged_program(2, 4, 3)
    fused = build_scan_fused_program(2, 4, 3)
    assert staged.inputs == fused.inputs == ["w", "u", "k", "decay"]
    assert staged.outputs == fused.outputs == ["h_out", "final"]
    # the staged lowering unrolls nc chunks into many nodes; the fused one is single.
    assert len(staged.nodes) > len(fused.nodes)


def test_fused_registry_contracts():
    reg = default_fused_registry()
    scan = reg.contract("chunk_h_scan")
    assert scan.in_slots == ["w", "u", "k", "decay"]
    assert scan.out_names == ["h_out", "final"]
    kkt = reg.contract("kkt_gated")
    assert kkt.in_slots == ["k", "g_sum", "beta"] and kkt.out_names == ["L"]
    with pytest.raises(KeyError):
        reg.kernel("nope", {})


def test_kkt_fused_program_single_node():
    prog = build_kkt_fused_program(4, 8)
    assert isinstance(prog.nodes[0], FusedNode)
    assert prog.outputs == ["L"]


def test_decision_kept_only_if_gated_det_and_faster():
    keep = FusionDecision("s", "k", gated_ok=True, deterministic=True, faster=True,
                          kept=True, t_staged_ms=1.0, t_fused_ms=0.5, frob=0.0, detail="")
    assert keep.kept and abs(keep.speedup - 2.0) < 1e-9 and "FUSE" in str(keep)
    # any failing gate => the staged lowering stands
    drop = FusionDecision("s", "k", gated_ok=True, deterministic=False, faster=True,
                          kept=False, t_staged_ms=1.0, t_fused_ms=0.5, frob=0.0, detail="")
    assert not drop.kept and "stage" in str(drop)
    text = format_fusion_decisions([keep, drop])
    assert "1/2 stages fused" in text
