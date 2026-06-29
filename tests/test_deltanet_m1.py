"""M1 exit test: the DeltaNet forward runs from IR, staged, gated green.

Requires an NPU (compiles + runs real substrate kernels). The CI shape is small
for speed; the M=16384 exit shape is exercised by `run_deltanet.py`.
"""
import os

import pytest
import torch

pytest.importorskip("torch_npu")
import torch_npu  # noqa

from pto_fuser import StagedExecutor, gate_determinism, gate_outputs
from pto_fuser.forwards import (build_deltanet_program, deltanet_reference,
                                make_inputs)

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="DeltaNet M1 test needs an Ascend NPU")


@pytest.mark.parametrize("B,H,nc,C,D", [(1, 2, 3, 64, 128)])
def test_deltanet_from_ir_staged_gated(B, H, nc, C, D):
    torch.npu.set_device("npu:0")
    dev = torch.device("npu:0")
    torch.manual_seed(0)
    scale = D ** -0.5

    program = build_deltanet_program(B, H, nc, C, D, scale)
    bindings = make_inputs(B, H, nc, C, D, dev)
    ref = deltanet_reference(**bindings, B=B, H=H, nc=nc, C=C, D=D, scale=scale)

    ex = StagedExecutor()
    got = ex.run(program, bindings)

    # frob_rel gate, every stage (same 2e-2 tolerance as the prototype e2e).
    results = gate_outputs(got, ref, tol=2e-2)
    failed = [str(r) for r in results if not r.passed]
    assert not failed, "frob_rel gate failed:\n" + "\n".join(failed)

    # determinism gate: bit-identical across two runs (no cumsum/mega here).
    det = gate_determinism(lambda: ex.run(program, bindings))
    assert det.passed, str(det)
