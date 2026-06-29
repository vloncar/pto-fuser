"""M2 exit test (NPU): the Planner measures levers 2/3 on the DeltaNet and GDN
contraction stages, and the lever-pinned program stays gated-green.

Asserts the planner's contract, not a fixed speedup (timings vary by machine):
  * every decision is self-consistent — kept iff fired & gated & faster;
  * the substrate's direct-read lowering is bit-equivalent to the Phase-A baseline
    on every stage (gated_ok), which is the correctness half of the lever;
  * the annotated DeltaNet program still matches the fp32 reference.
"""
import pytest
import torch

pytest.importorskip("torch_npu")
import torch_npu  # noqa

from pto_fuser import Planner, StagedExecutor, gate_outputs
from pto_fuser.forwards import (build_deltanet_program, deltanet_reference,
                                gdn_contraction_stages, make_inputs)

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="M2 planner test needs an Ascend NPU")


def _check_consistency(decisions):
    for d in decisions:
        assert d.kept == (d.fired and d.gated_ok and d.faster), str(d)


def test_planner_gdn_stages_gated_and_consistent():
    torch.npu.set_device("npu:0")
    torch.manual_seed(0)
    planner = Planner()
    fired_modes = set()
    for name, prog, bindings in gdn_contraction_stages(B=1, nc=2, H=16, C=64, D=128):
        _, decisions = planner.plan(prog, bindings)
        _check_consistency(decisions)
        # the direct-read lowering must be correctness-equivalent to baseline.
        dr = [d for d in decisions if d.lever == "direct_read"][0]
        assert dr.gated_ok, f"{name}: direct-read diverged from baseline: {dr}"
        if dr.fired:
            fired_modes.add(dr.detail.split()[0])
    # the GDN family should exercise more than one direct-read mode (NT/NN/TN).
    assert len(fired_modes) >= 2, f"expected multiple read modes, saw {fired_modes}"


def test_planner_deltanet_annotated_program_gates_green():
    torch.npu.set_device("npu:0")
    dev = torch.device("npu:0")
    torch.manual_seed(0)
    B, H, nc, C, D = 1, 2, 3, 64, 128
    scale = D ** -0.5

    program = build_deltanet_program(B, H, nc, C, D, scale)
    bindings = make_inputs(B, H, nc, C, D, dev)
    ref = deltanet_reference(**bindings, B=B, H=H, nc=nc, C=C, D=D, scale=scale)

    annotated, decisions = Planner().plan(program, bindings)
    _check_consistency(decisions)

    got = StagedExecutor().run(annotated, bindings)
    bad = [str(r) for r in gate_outputs(got, ref, tol=2e-2) if not r.passed]
    assert not bad, "annotated program diverged from reference:\n" + "\n".join(bad)
