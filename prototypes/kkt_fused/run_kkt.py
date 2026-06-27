"""Build, validate, and time the fused-kkt prototype (V1, two-pass).

  python run_kkt.py --nc 2 --H 4            # quick correctness
  python run_kkt.py --nc 512 --H 32 --time  # full 8x8192 kkt + timing
"""
import argparse, ctypes, hashlib, os, subprocess, sys
import torch, torch_npu  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
# pto-fuser depends on pto-einsum (sibling repo by default; override with PTO_EINSUM).
REPO = os.environ.get("PTO_EINSUM", os.path.abspath(os.path.join(HERE, "..", "..", "..", "pto-einsum")))
INC = os.path.join(REPO, "src", "pto_einsum", "include")
C = 128; D = 128


def compile_lib(nc, H):
    ascend = os.environ["ASCEND_HOME_PATH"]
    pto = os.environ["PTO_LIB_PATH"]
    arch = os.environ.get("NPU_ARCH", "dav-2201")
    defs = os.environ.get("KKT_DEFS", "").split()
    tag = hashlib.md5(f"{nc}_{H}_{' '.join(defs)}".encode()).hexdigest()[:10]
    so = os.path.join(HERE, f"kkt_{tag}.so")
    if not os.path.exists(so):
        cmd = ["bisheng", "-O3", "-shared", "-fPIC", "-std=c++17", "-xcce",
               f"--npu-arch={arch}", f"-DKKT_NC={nc}", f"-DKKT_H={H}", *defs,
               "-I", HERE, "-I", INC, "-I", f"{ascend}/include", "-I", f"{pto}/include",
               "-L", f"{ascend}/lib64", "-lascendcl", "-lruntime",
               os.path.join(HERE, "kkt_fused_lib.cpp"), "-o", so]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr); sys.exit(1)
    lib = ctypes.CDLL(so)
    lib.kkt_setup.restype = ctypes.c_void_p
    lib.kkt_exec.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_int32, ctypes.c_int64, ctypes.c_void_p]
    lib.kkt_teardown.argtypes = [ctypes.c_void_p]
    lib.kkt_setup_v2.restype = ctypes.c_void_p
    lib.kkt_exec_v2.argtypes = lib.kkt_exec.argtypes
    return lib


def vp(t): return ctypes.c_void_p(t.data_ptr())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", type=int, default=2)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--Hg", type=int, default=None)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--v2", action="store_true", help="use the per-tile interleave kernel")
    a = ap.parse_args()
    nc, H = a.nc, a.H
    Hg = a.Hg if a.Hg else H // 2
    grp = H // Hg
    T = nc * C
    dev = "npu:0"
    torch.manual_seed(0)

    k = torch.nn.functional.normalize(torch.randn(1, T, Hg, D), dim=-1).to(dev).half()
    beta = torch.rand(1, T, H, device=dev).half()
    g_in = torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=dev).float())
    g_sum = torch.zeros_like(g_in)
    for j in range(0, T, C):
        g_sum[:, j:j+C] = g_in[:, j:j+C].cumsum(1)

    kc = k.repeat_interleave(grp, dim=2)                      # [1,T,H,D] half
    kc_flat = kc.reshape(nc, C, H, D).contiguous().reshape(-1, H, D).contiguous()
    g_t = g_sum[0].permute(1, 0).contiguous()                # [H,T] f32
    beta_t = beta[0].permute(1, 0).contiguous()              # [H,T] half
    mask = (torch.arange(C, device=dev)[:, None] > torch.arange(C, device=dev)[None, :]).float()

    L = torch.zeros(T, H, C, device=dev).half()
    lib = compile_lib(nc, H)
    setup = lib.kkt_setup_v2 if a.v2 else lib.kkt_setup
    exec_ = lib.kkt_exec_v2 if a.v2 else lib.kkt_exec
    ws = setup()
    stream = torch.npu.current_stream()._as_parameter_
    exec_(vp(kc_flat), vp(g_t), vp(beta_t), vp(mask), ctypes.c_void_p(ws),
          vp(L), ctypes.c_int32(H), ctypes.c_int64(T), stream)
    torch.npu.synchronize()

    # torch reference (exactly what the kernel computes)
    kcr = kc.reshape(nc, C, H, D).float()
    qk = torch.einsum("cihd,cjhd->cihj", kcr, kcr)           # [nc,C,H,C]
    gs = g_sum[0].reshape(nc, C, H)
    bs = beta[0].reshape(nc, C, H).float()
    gv = gs + torch.log(bs)
    coeff = torch.exp(torch.clamp(gv.unsqueeze(2) - gs.unsqueeze(1), max=0.0))  # [nc,i,j,h]? fix dims
    # gv[c,i,h]-gs[c,j,h] -> want [nc,C(i),H,C(j)]
    diff = gv.permute(0,2,1).unsqueeze(-1) - gs.permute(0,2,1).unsqueeze(2)     # [nc,H,i,j]
    coeff = torch.exp(torch.clamp(diff, max=0.0)).permute(0,2,1,3)              # [nc,i,H,j]
    m = (torch.arange(C)[:, None] > torch.arange(C)[None, :]).float().to(dev)
    if os.environ.get("KKT_DEFS", "").find("KKT_PLAIN") >= 0:
        Lref = qk.reshape(T, H, C).half()   # plain: raw qk, no gating/mask
    else:
        Lref = (qk * coeff * m[None,:,None,:]).reshape(T, H, C).half()

    err = (L.float() - Lref.float())
    rel = err.norm() / Lref.float().norm().clamp_min(1e-9)
    amax = err.abs().max().item()
    print(f"nc={nc} H={H} Hg={Hg} T={T} I={nc*H}  frob_rel={rel:.3e}  max_abs={amax:.4e}  "
          f"{'OK' if rel < 2e-2 else 'FAIL'}")

    if a.time:
        import time as _t
        ITERS = 50
        for _ in range(5):
            exec_(vp(kc_flat), vp(g_t), vp(beta_t), vp(mask), ctypes.c_void_p(ws),
                  vp(L), ctypes.c_int32(H), ctypes.c_int64(T), stream)
        torch.npu.synchronize()
        t0 = _t.perf_counter()
        for _ in range(ITERS):
            exec_(vp(kc_flat), vp(g_t), vp(beta_t), vp(mask), ctypes.c_void_p(ws),
                  vp(L), ctypes.c_int32(H), ctypes.c_int64(T), stream)
        torch.npu.synchronize()
        ms = (_t.perf_counter() - t0) * 1000 / ITERS
        tag = "v2 (interleave)" if a.v2 else "v1 (two-pass)"
        print(f"  fused kkt {tag} mean: {ms:.3f} ms  "
              f"(pto-einsum staged kkt @8x8192: 36.8 ms; megagdn: 2.99 ms)")
    lib.kkt_teardown(ws)


if __name__ == "__main__":
    main()
