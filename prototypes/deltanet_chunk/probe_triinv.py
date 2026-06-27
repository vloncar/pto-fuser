"""T3 step 0 -- de-risk the opaque node.

Pin down the exact input/output CONTRACT of the hand-optimized triangular-inverse
kernel (pto-kernels rec_unroll) when invoked from OUR workspace, before composing
it into the DeltaNet chunk pipeline. This is the genuinely-new T3 question: can the
fusion layer host an opaque kernel that is NOT the einsum tile-matmul-core+epilogue
shape? Step 0 just establishes the callable + its triangular/transpose convention.

We reuse pto-kernels' own proven JIT build (fast_inverse.cpp + jit_util) rather than
re-deriving the build flags -- it compiles for Ascend910B4 CubeCore, which is the
SAME a2/a3 family as our dav-2201 einsum substrate (see arch-dav-family-map memory),
so a staged (multi-launch) composition only needs the two stages to agree on the GM
tensor layout, not on build flags.

    export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
    python prototypes/deltanet_chunk/probe_triinv.py
"""
import os, sys
os.environ.setdefault("PTO_LIB_PATH", "/home/vloncar/work/einsum_workspace/pto-isa")

import torch, torch_npu  # noqa

# Reuse the pto-kernels JIT build for the opaque tri-inv node.
PTO_KERNELS = "/home/vloncar/work/einsum_workspace/pto-kernels"
FAST_INV_DIR = os.path.join(PTO_KERNELS, "examples", "jit_cpp", "fast_inverse")
sys.path.insert(0, FAST_INV_DIR)
from jit_util_fast_inverse import jit_compile  # noqa


def make_minus_identity(C, device):
    mi = torch.zeros((C, C), dtype=torch.float16, device=device)
    mi.fill_diagonal_(-1)
    return mi


def main():
    torch.npu.set_device("npu:0")
    dev = torch.device("npu:0")
    torch.manual_seed(0)

    tri_inv = jit_compile(os.path.join(FAST_INV_DIR, "fast_inverse.cpp"), verbose=True)

    C = 64            # chunk size / matrix side (must be in {16,32,64,128})
    nmat = 8          # a few independent matrices, consecutive (non-BSND) layout

    # Build strictly-LOWER unit-triangular A (diagonal implied = 1, kernel adds it).
    # Keep ||A|| modest so fp16 (I+A)^-1 is well-conditioned.
    A = (torch.randn(nmat, C, C, dtype=torch.float16, device=dev) * 0.05)
    A = torch.tril(A, diagonal=-1)            # strictly lower, zero diagonal

    # Clean fp64 reference: invert (I + A) per matrix.
    eye = torch.eye(C, dtype=torch.float64, device="cpu")
    A64 = A.double().cpu()
    ref = torch.linalg.solve_triangular(
        (eye[None] + A64), eye[None].expand(nmat, C, C), upper=False, unitriangular=True
    )  # (I+A)^-1, lower-triangular

    out = torch.zeros_like(A, dtype=torch.float32)
    mi = make_minus_identity(C, dev)
    torch.npu.synchronize()
    tri_inv(out, A, mi, C, nmat, 0, cu_seqlens=None,
            block_dim=min(20, nmat))  # is_lower handled internally? probe below
    torch.npu.synchronize()

    got = out.double().cpu()
    err = (got - ref)
    rel = err.norm() / ref.norm().clamp_min(1e-9)
    print(f"\n  [lower-as-is]  frob_rel={rel:.3e}  max_abs={err.abs().max():.3e}")

    # The kernel default is UPPER-triangular; lower needs the transpose convention
    # the varlen example uses. Probe the transposed contract too.
    out2 = torch.zeros_like(A, dtype=torch.float32)
    A_T = A.transpose(-1, -2).contiguous()
    torch.npu.synchronize()
    tri_inv(out2, A_T, mi, C, nmat, 0, cu_seqlens=None, block_dim=min(20, nmat))
    torch.npu.synchronize()
    got2 = out2.double().cpu().transpose(-1, -2)
    err2 = (got2 - ref)
    rel2 = err2.norm() / ref.norm().clamp_min(1e-9)
    print(f"  [upper-via-transpose]  frob_rel={rel2:.3e}  max_abs={err2.abs().max():.3e}")


if __name__ == "__main__":
    main()
