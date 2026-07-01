"""End-to-end GDN benchmark: the pto-fuser gated forward vs the megagdn megakernel.

The full equivalent of ``pto-einsum/benchmarks/complex/gdn/bench_gdn_4way.py``, but the
contraction/glue side is the **fusion layer**: the complete gated GDN forward built as a
fuser `Program` (``attention/_gdn_full``), graph-captured, put head-to-head with the
megagdn megakernel across a grid of head counts and sequence lengths. megagdn is the
baseline (1.00×); both are gated (Frobenius) against the fp32 reference.

Simplification: ``Hg = H`` (no GQA); megagdn is run with ``key_heads = H`` to match.

    export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
    python examples/benchmarks/gdn_mega.py --configs 16x4,32x8,64x4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # benchmarks/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # examples/
import common  # noqa: E402

import torch  # noqa: E402

from attention._gdn_full import (build_gdn_program, gdn_reference,  # noqa: E402
                                 make_gdn_inputs, prepare_gdn_bindings)
import _mega_bench as mb  # noqa: E402

# megagdn-pto is a sibling repo; its staged adapter lives in the pto-einsum benchmark.
_MEGA_ADAPTER = os.path.join(os.path.dirname(__file__),
                             "../../../pto-einsum/benchmarks/complex/gdn")


def _mega_layout(inp):
    """Our [1,H,nc,C,D] / [1,H,nc,C] operands -> megagdn [1,T,H,D] / [1,T,H]."""
    q, k, v, beta, g_in = (inp[n] for n in ("q", "k", "v", "beta", "g_in"))
    B, H, nc, C, D = q.shape
    T = nc * C
    to_thd = lambda t: t.permute(0, 2, 3, 1, 4).reshape(1, T, H, D).contiguous()
    to_th = lambda t: t.permute(0, 2, 3, 1).reshape(1, T, H).contiguous()
    return to_thd(q), to_thd(k), to_thd(v), to_th(beta), to_th(g_in.to(q.device)), T


def _mega_runner(inp, B, H, nc, C, D, scale, dev):
    sys.path.insert(0, os.path.abspath(_MEGA_ADAPTER))
    from mega_gdn import MegaGDN  # noqa: E402
    q, k, v, beta, g_in = (inp[n] for n in ("q", "k", "v", "beta", "g_in"))
    qm, km, vm, bm, gm, T = _mega_layout(inp)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    mega = MegaGDN(qm, km, vm, gm.float(), bm, cu, H, H, scale)   # Hg = H
    return lambda: mega.run_full_pipeline()                       # -> [1,T,H,D]


def _to_mega_golden(ref):       # [B=1,H,nc,C,D] -> [1,T,H,D]
    B, H, nc, C, D = ref.shape
    return ref.permute(0, 2, 3, 1, 4).reshape(1, nc * C, H, D).contiguous()


FAMILY = type("Family", (), dict(
    make_inputs=staticmethod(make_gdn_inputs),
    reference=staticmethod(lambda inp, scale: gdn_reference(
        inp["q"], inp["k"], inp["v"], inp["beta"], inp["g_in"], scale)),
    build=staticmethod(build_gdn_program),
    prepare=staticmethod(lambda inp: prepare_gdn_bindings(
        inp["q"], inp["k"], inp["v"], inp["beta"], inp["g_in"])),
    mega_runner=staticmethod(_mega_runner),
    to_mega_golden=staticmethod(_to_mega_golden),
))()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="16x4,16x8,16x16,32x4,32x8,64x4,64x8",
                    help="comma list of HxNC (head count x chunk count), B=1, C=D=128, so "
                         "sequence length T = nc*128. megagdn supports H in {16,24,32,48,64}; "
                         "the fuser is head-agnostic.")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    dev = common.pick_device()
    if dev is None:
        print("no healthy NPU — this benchmark needs an Ascend device.")
        return
    configs = mb.parse_configs(args.configs,
                               [(16, 4), (16, 8), (16, 16), (32, 4), (32, 8), (64, 4), (64, 8)])
    mb.run(FAMILY, dev, configs, args.iters, os.path.dirname(os.path.abspath(__file__)),
           title="GDN — fuser forward vs megagdn", slug="gdn_mega")


if __name__ == "__main__":
    main()
