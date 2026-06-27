"""Build, validate, and time the T2 cross-chunk-scan prototype.

Tests the genuinely-new chunk-attention topology: a *sequential* recurrence over
chunks where the resident state S is BOTH a matmul operand (w@S) and a Vec-updated
accumulator (S = decay*S + k^T@vc). Parallel over (batch,head); sequential over
chunks within each (b,h). This is the dataflow the staged einsum path can't fuse
(per-chunk launch -> "launch-bound 5x"); the prototype keeps the loop in ONE kernel.

Reference recurrence (per b,h), our own clean target (GDN chunk_h minus per-token
coeff -- same topology):

    S = 0                              # [D,D]
    for c in chunks:
        h_out[c] = S                   # readout previous state
        WS = w[c] @ S                  # matmul reads resident S      (in_nt=2)
        vc = u[c] - WS                 # Vec residual
        S = decay[c]*S + k[c].T @ vc   # matmul-accumulate + recurrence (in_nt=3)

  python run_scan.py --B 1 --H 2 --nc 3      # quick correctness
  python run_scan.py --B 8 --H 32 --nc 64 --time
"""
import argparse, ctypes, hashlib, os, subprocess, sys
import torch, torch_npu  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
# pto-fuser depends on pto-einsum (sibling repo by default; override with PTO_EINSUM).
REPO = os.environ.get("PTO_EINSUM", os.path.abspath(os.path.join(HERE, "..", "..", "..", "pto-einsum")))
INC = os.path.join(REPO, "src", "pto_einsum", "include")
C = 128; D = 128


def compile_lib(B, H, nc):
    ascend = os.environ["ASCEND_HOME_PATH"]
    pto = os.environ["PTO_LIB_PATH"]
    arch = os.environ.get("NPU_ARCH", "dav-2201")
    defs = os.environ.get("SCAN_DEFS", "").split()
    tag = hashlib.md5(f"{B}_{H}_{nc}_{' '.join(defs)}".encode()).hexdigest()[:10]
    so = os.path.join(HERE, f"scan_{tag}.so")
    if not os.path.exists(so):
        cmd = ["bisheng", "-O3", "-shared", "-fPIC", "-std=c++17", "-xcce",
               f"--npu-arch={arch}", f"-DSCAN_B={B}", f"-DSCAN_H={H}", f"-DSCAN_NC={nc}", *defs,
               "-I", HERE, "-I", INC, "-I", f"{ascend}/include", "-I", f"{pto}/include",
               "-L", f"{ascend}/lib64", "-lascendcl", "-lruntime",
               os.path.join(HERE, "scan_lib.cpp"), "-o", so]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr); sys.exit(1)
    lib = ctypes.CDLL(so)
    lib.scan_setup.restype = ctypes.c_void_p
    # exec(w, u, k, decay, ws, h_out, final, stream)
    lib.scan_exec.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_void_p, ctypes.c_void_p,
                                                    ctypes.c_void_p, ctypes.c_void_p]
    lib.scan_teardown.argtypes = [ctypes.c_void_p]
    return lib


def vp(t): return ctypes.c_void_p(t.data_ptr())


def reference(w, u, k, decay, B, H, nc):
    """Clean fp32 reference of the simplified chunk_h recurrence."""
    dev = w.device
    h_out = torch.zeros(B, nc, H, D, D, device=dev)
    final = torch.zeros(B, H, D, D, device=dev)
    wf, uf, kf, df = w.float(), u.float(), k.float(), decay.float()
    for b in range(B):
        for h in range(H):
            S = torch.zeros(D, D, device=dev)
            for c in range(nc):
                h_out[b, c, h] = S
                WS = wf[b, c, :, h, :] @ S            # [C,D]
                vc = uf[b, c, :, h, :] - WS           # [C,D]
                S = df[b, h, c] * S + kf[b, c, :, h, :].T @ vc   # [D,D]
            final[b, h] = S
    return h_out, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    B, H, nc = a.B, a.H, a.nc
    dev = "npu:0"
    torch.manual_seed(0)

    # Operands token-major [B, nc, C, H, D]. The per-chunk state map is
    # (decay*I - k^T w) S + k^T u; keep ||k^T w|| and decay small enough that the
    # operator is contractive, else fp16 overflows over nc chunks (synthetic-test
    # stability only -- real GDN stays bounded via the per-token decay we dropped).
    w = (torch.randn(B, nc, C, H, D, device=dev) * 0.02).half()
    k = (torch.randn(B, nc, C, H, D, device=dev) * 0.02).half()
    u = (torch.randn(B, nc, C, H, D, device=dev) * 0.05).half()
    decay = (0.7 + 0.2 * torch.rand(B, H, nc, device=dev)).float()   # per-chunk scalar in [0.7,0.9)

    h_out = torch.zeros(B, nc, H, D, D, device=dev).half()
    final = torch.zeros(B, H, D, D, device=dev).half()

    lib = compile_lib(B, H, nc)
    ws = lib.scan_setup()
    stream = torch.npu.current_stream()._as_parameter_
    lib.scan_exec(vp(w), vp(u), vp(k), vp(decay), ctypes.c_void_p(ws),
                  vp(h_out), vp(final), stream)
    torch.npu.synchronize()

    h_ref, f_ref = reference(w, u, k, decay, B, H, nc)
    for name, got, ref in [("h_out", h_out, h_ref), ("final", final, f_ref)]:
        err = (got.float() - ref.float())
        rel = err.norm() / ref.float().norm().clamp_min(1e-9)
        amax = err.abs().max().item()
        print(f"  {name:6s} frob_rel={rel:.3e}  max_abs={amax:.4e}  "
              f"{'OK' if rel < 2e-2 else 'FAIL'}")

    if a.time:
        import time as _t
        ITERS = 50
        for _ in range(5):
            lib.scan_exec(vp(w), vp(u), vp(k), vp(decay), ctypes.c_void_p(ws),
                          vp(h_out), vp(final), stream)
        torch.npu.synchronize()
        t0 = _t.perf_counter()
        for _ in range(ITERS):
            lib.scan_exec(vp(w), vp(u), vp(k), vp(decay), ctypes.c_void_p(ws),
                          vp(h_out), vp(final), stream)
        torch.npu.synchronize()
        ms = (_t.perf_counter() - t0) * 1000 / ITERS
        print(f"  scan B={B} H={H} nc={nc} mean: {ms:.3f} ms")
    lib.scan_teardown(ws)


if __name__ == "__main__":
    main()
