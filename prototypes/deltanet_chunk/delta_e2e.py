"""T3 e2e -- full DeltaNet chunked forward composed on the substrate.

The structural study for the auto-fuser vision: confirm the COMPLETE DeltaNet
forward decomposes entirely into substrate primitives + the one opaque node, with
no missing capability and no surprises at the stage boundaries. Mirrors the fla
chunk_gated_delta_rule_fwd pipeline (megagdn-pto/.../fla_vendor/chunk.py):

    1. kkt        A   = tril( (beta k) @ k^T , -1)        einsum-core  "nid,njd->nij"
    2. solve_tril T   = (I + A)^-1                         OPAQUE NODE  (rec_unroll)
    3. recompute  W   = T @ (beta k)                       einsum-core  "nij,njd->nid"
                  U   = T @ (beta v)                       einsum-core  "nij,njd->nid"
    4. scan       per chunk c, state S [D,D]:              einsum-core  (sequential)
                    h_state[c] = S
                    v_new[c]   = U[c] - W[c] @ S           "nid,nde->nie"
                    S          = S + k[c]^T @ v_new[c]     "nid,nie->nde"
    5. chunk_o    o[c] = q[c]@h_state[c]                   "nid,nde->nie"
                       + tril(q[c]@k[c]^T,0) @ v_new[c]    "nid,njd->nij" + "nij,nje->nie"

Pure DeltaNet (gate g = 0) -- the per-element exp(g) terms in the gated variant are
Vec scalings (a proven primitive class, exercised in kkt_fused / chunk_h_scan), so
dropping them loses no STRUCTURAL coverage while removing a large convention-bug
surface. This isolates the only thing the e2e is meant to test: does the full
pipeline COMPOSE across every stage boundary, bit-faithfully, through the substrate?

Reference = the identical pipeline in fp32 torch; we validate substrate-vs-fp32 at
every stage, so any boundary/layout/marshalling fault (e.g. the NZ-vs-ND minefield)
shows up as a stage that diverges. The cross-chunk scan is run as a staged per-chunk
einsum loop (the chunk_h_scan prototype already proved the fused single-kernel form;
here we only need the dataflow to compose).

    export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
    python prototypes/deltanet_chunk/delta_e2e.py --B 1 --H 2 --nc 3 --C 64
    python prototypes/deltanet_chunk/delta_e2e.py --B 8 --H 32 --nc 64 --C 64
"""
import argparse, os, sys
os.environ.setdefault("PTO_LIB_PATH", "/home/vloncar/work/einsum_workspace/pto-isa")

import torch, torch_npu  # noqa
# pto-fuser depends on pto-einsum (sibling repo by default; override with PTO_EINSUM).
_EINSUM = os.environ.get(
    "PTO_EINSUM", os.path.join(os.path.dirname(__file__), "..", "..", "..", "pto-einsum"))
sys.path.insert(0, os.path.join(_EINSUM, "src"))
from pto_einsum import einsum  # noqa

PTO_KERNELS = os.environ.get("PTO_KERNELS", "/home/vloncar/work/einsum_workspace/pto-kernels")
FAST_INV_DIR = os.path.join(PTO_KERNELS, "examples", "jit_cpp", "fast_inverse")
sys.path.insert(0, FAST_INV_DIR)
from jit_util_fast_inverse import jit_compile  # noqa


def opaque_tri_inv(tri_inv, A_lower):
    """Invert (I + A_lower) per matrix via the opaque rec_unroll kernel.

    Contract (pinned in probe_triinv.py): the kernel inverts strictly-UPPER unit
    matrices, so feed A^T (strictly upper) and transpose the fp32 result back.
    A_lower: [M, C, C] fp16, strictly lower (zero diagonal). The raw <<<>>> launch
    needs an explicit sync on each side to order against the torch stream."""
    M, C, _ = A_lower.shape
    A_upper = A_lower.transpose(-1, -2).contiguous()
    out = torch.zeros_like(A_upper, dtype=torch.float32)
    mi = torch.zeros((C, C), dtype=torch.float16, device=A_lower.device)
    mi.fill_diagonal_(-1)
    torch.npu.synchronize()
    tri_inv(out, A_upper, mi, C, M, 0, cu_seqlens=None, block_dim=min(20, M))
    torch.npu.synchronize()
    return out.transpose(-1, -2).contiguous()   # (I+A)^-1, lower, fp32


def es(eq, a, b):
    """einsum on the substrate, cast fp32->fp16.

    CORRECTION (T3 e2e): the einsum runner returns a PLAIN torch.empty fp32 tensor
    (base ND, no padding -- the earlier "2x NZ-padding minefield" was a misread of
    fp32 byte size vs an fp16 expectation). The ONLY boundary requirement is the
    fp32->fp16 cast the opaque kernel needs; a bare .half() suffices. The raw <<<>>>
    opaque launch still needs the syncs in opaque_tri_inv (ctypes-launch ordering vs
    the torch stream), but there is no layout laundering to do."""
    return einsum(eq, a.half(), b.half()).half()


# --------------------------------------------------------------------------- #
#  Reference: the identical pipeline in fp32 torch.
# --------------------------------------------------------------------------- #
def reference(q, k, v, beta, B, H, nc, C, D, scale):
    qf, kf, vf, bf = q.float(), k.float(), v.float(), beta.float()
    kb = bf[..., None] * kf                                   # [B,H,nc,C,D]
    vb = bf[..., None] * vf
    flat = lambda t: t.reshape(B * H * nc, C, -1)
    kbF, vbF, kF, qF = flat(kb), flat(vb), flat(kf), flat(qf)
    Araw = torch.einsum("nid,njd->nij", kbF, kF)
    A = torch.tril(Araw, diagonal=-1)
    eye = torch.eye(C, device=q.device).expand(B * H * nc, C, C)
    T = torch.linalg.solve_triangular(eye + A, eye, upper=False, unitriangular=True)
    W = torch.matmul(T, kbF).reshape(B, H, nc, C, D)
    U = torch.matmul(T, vbF).reshape(B, H, nc, C, D)
    kf5 = kf.reshape(B, H, nc, C, D)
    # cross-chunk scan
    h_state = torch.zeros(B, H, nc, D, D, device=q.device)
    v_new = torch.zeros(B, H, nc, C, D, device=q.device)
    for b in range(B):
        for h in range(H):
            S = torch.zeros(D, D, device=q.device)
            for c in range(nc):
                h_state[b, h, c] = S
                vn = U[b, h, c] - W[b, h, c] @ S
                v_new[b, h, c] = vn
                S = S + kf5[b, h, c].transpose(-1, -2) @ vn
    # output
    o_inter = torch.einsum("nid,nde->nie", qF, h_state.reshape(-1, D, D))
    Aqk = torch.tril(torch.einsum("nid,njd->nij", qF, kF), diagonal=0)
    o_intra = torch.einsum("nij,nje->nie", Aqk, v_new.reshape(-1, C, D))
    o = (o_inter + o_intra) * scale
    return dict(A=A, T=T, W=W, U=U, h_state=h_state, v_new=v_new,
                o=o.reshape(B, H, nc, C, D))


# --------------------------------------------------------------------------- #
#  Composition: same pipeline, substrate einsum + opaque node.
# --------------------------------------------------------------------------- #
def composed(q, k, v, beta, tri_inv, B, H, nc, C, D, scale):
    flat = lambda t: t.reshape(B * H * nc, C, -1).contiguous()
    kb = (beta[..., None] * k).half()
    vb = (beta[..., None] * v).half()
    kbF, vbF, kF, qF = flat(kb), flat(vb), flat(k.half()), flat(q.half())

    # 1. kkt  ->  2. opaque inverse
    Araw = es("nid,njd->nij", kbF, kF)                       # [M,C,C]
    A = torch.tril(Araw, diagonal=-1).contiguous()
    T = opaque_tri_inv(tri_inv, A).half()                   # [M,C,C]

    # 3. recompute W, U  (two einsums off the same T)
    W = es("nij,njd->nid", T, kbF).reshape(B, H, nc, C, D)
    U = es("nij,njd->nid", T, vbF).reshape(B, H, nc, C, D)
    k5 = kF.reshape(B, H, nc, C, D)

    # 4. scan (staged per-chunk einsum loop; parallel over BH, sequential over c)
    BH = B * H
    Wb = W.reshape(BH, nc, C, D); Ub = U.reshape(BH, nc, C, D)
    kb5 = k5.reshape(BH, nc, C, D)
    h_state = torch.zeros(BH, nc, D, D, device=q.device).half()
    v_new = torch.zeros(BH, nc, C, D, device=q.device).half()
    S = torch.zeros(BH, D, D, device=q.device).half()
    for c in range(nc):
        h_state[:, c] = S
        WS = es("nid,nde->nie", Wb[:, c].contiguous(), S)   # W[c] @ S
        vn = (Ub[:, c].float() - WS.float()).half()
        v_new[:, c] = vn
        dS = es("nid,nie->nde", kb5[:, c].contiguous(), vn) # k[c]^T @ v_new[c]
        S = (S.float() + dS.float()).half()

    # 5. chunk_o
    h_flat = h_state.reshape(B * H * nc, D, D).contiguous()
    vn_flat = v_new.reshape(B * H * nc, C, D).contiguous()
    o_inter = es("nid,nde->nie", qF, h_flat)
    Aqk = torch.tril(es("nid,njd->nij", qF, kF), diagonal=0).contiguous()
    o_intra = es("nij,nje->nie", Aqk, vn_flat)
    o = ((o_inter.float() + o_intra.float()) * scale).reshape(B, H, nc, C, D)
    return dict(A=A.reshape(B * H * nc, C, C), T=T, W=W, U=U,
                h_state=h_state.reshape(B, H, nc, D, D),
                v_new=v_new.reshape(B, H, nc, C, D), o=o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--C", type=int, default=64, help="chunk size in {16,32,64,128}")
    ap.add_argument("--D", type=int, default=128)
    a = ap.parse_args()
    torch.npu.set_device("npu:0")
    dev = torch.device("npu:0")
    torch.manual_seed(0)
    B, H, nc, C, D = a.B, a.H, a.nc, a.C, a.D
    scale = D ** -0.5

    tri_inv = jit_compile(os.path.join(FAST_INV_DIR, "fast_inverse.cpp"), verbose=False)

    nrm = torch.nn.functional.normalize
    # small magnitudes keep (I+A)^-1 well-conditioned and the scan contractive in fp16
    k = nrm(torch.randn(B, H, nc, C, D, device=dev, dtype=torch.float16), dim=-1)
    q = nrm(torch.randn(B, H, nc, C, D, device=dev, dtype=torch.float16), dim=-1)
    v = (torch.randn(B, H, nc, C, D, device=dev, dtype=torch.float16) * 0.1)
    beta = torch.rand(B, H, nc, C, device=dev, dtype=torch.float16) * 0.5

    ref = reference(q, k, v, beta, B, H, nc, C, D, scale)
    got = composed(q, k, v, beta, tri_inv, B, H, nc, C, D, scale)

    print(f"  DeltaNet e2e  B={B} H={H} nc={nc} C={C} D={D}  (M={B*H*nc} chunks)")
    ok = True
    for name in ["A", "T", "W", "U", "v_new", "h_state", "o"]:
        g = got[name].float(); r = ref[name].float()
        err = g - r
        rel = (err.norm() / r.norm().clamp_min(1e-9)).item()
        amax = err.abs().max().item()
        flag = "OK" if rel < 2e-2 else "FAIL"
        ok = ok and rel < 2e-2
        print(f"    {name:8s} frob_rel={rel:.3e}  max_abs={amax:.3e}  {flag}")
    print("  ALL OK" if ok else "  *** FAIL ***")


if __name__ == "__main__":
    main()
