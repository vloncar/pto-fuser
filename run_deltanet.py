"""M1 driver: run the DeltaNet forward from its IR Program, staged, and gate it.

    export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
    python run_deltanet.py --B 1 --H 2 --nc 3 --C 64        # quick
    python run_deltanet.py --B 8 --H 32 --nc 64 --C 64      # M1 exit shape (M=16384)
"""
import argparse
import os
import sys

os.environ.setdefault("PTO_LIB_PATH", "/home/vloncar/work/einsum_workspace/pto-isa")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch_npu  # noqa

from pto_fuser import StagedExecutor, gate_determinism, gate_outputs
from pto_fuser.forwards import (build_deltanet_program, deltanet_reference,
                                make_inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--tol", type=float, default=2e-2)
    a = ap.parse_args()

    torch.npu.set_device("npu:0")
    dev = torch.device("npu:0")
    torch.manual_seed(0)
    B, H, nc, C, D = a.B, a.H, a.nc, a.C, a.D
    scale = D ** -0.5

    program = build_deltanet_program(B, H, nc, C, D, scale)
    bindings = make_inputs(B, H, nc, C, D, dev)
    ref = deltanet_reference(**bindings, B=B, H=H, nc=nc, C=C, D=D, scale=scale)

    ex = StagedExecutor()
    got = ex.run(program, bindings)

    M = B * H * nc
    print(f"  DeltaNet (from IR)  B={B} H={H} nc={nc} C={C} D={D}  (M={M}, "
          f"{len(program.nodes)} nodes)")
    results = gate_outputs(got, ref, a.tol)
    for r in results:
        print("    " + str(r))
    det = gate_determinism(lambda: ex.run(program, bindings))
    print("    " + str(det))

    ok = all(r.passed for r in results) and det.passed
    print("  ALL OK" if ok else "  *** FAIL ***")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
