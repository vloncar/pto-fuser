"""Fused-kernel registry — the fused-node backend (see DESIGN.md).

A ``FusedNode`` names a hand-fused kernel that subsumes a staged sub-chain into one
dispatch, keeping intermediates on-chip. The registry maps that name to a recipe
that (lazily) compiles the prototype ``.cpp`` into a cached ``.so`` and to a
persistent runner: ``setup`` allocates the on-chip-state workspace once, ``run``
launches the kernel (no host sync — so the node is graph-capturable), and the
runner is reused across calls. Operand layout/dtype adapters and output allocation
live in the registered lowering, exactly as the opaque registry does for tri_inv.

Two seed entries, both proven prototypes built as their own ``.so`` sharing GM with
the surrounding stages (the design: *this* form works today; single-``.so`` opaque
inline is the unproven part and is not used here):

  * ``chunk_h_scan`` — the resident-state recurrence. State ``S`` stays
    resident across chunks (matmul operand + Vec accumulator); the staged lowering
    round-trips ``S`` through HBM every chunk.
  * ``kkt_gated``    — the matmul-core + on-chip gated/masked epilogue (glue absorption /
    glue absorption). The qk intermediate never lands in HBM.

Compile-time shape (``KKT_NC``/``KKT_H`` + ``KKT_C``/``KKT_D``; ``SCAN_B``/``SCAN_H``/
``SCAN_NC``) keys the ``.so``. The native-kkt path parameterizes the chunk/head dims
(``KKT_C``/``KKT_D``, default 128) so the zoo mechanisms run at their own ``C``/``D``
(e.g. ``C = 16, D = 64``); ``scan`` is still ``C = D = 128``.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_KERNELS = os.path.join(HERE, "kernels")
KKT_C = KKT_D = 128
SCAN_C = SCAN_D = 128


def _einsum_include() -> str:
    root = os.environ.get("PTO_EINSUM", os.path.abspath(
        os.path.join(HERE, "..", "..", "..", "pto-einsum")))
    return os.path.join(root, "src", "pto_einsum", "include")


def _vp(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _stream() -> ctypes.c_void_p:
    return torch.npu.current_stream()._as_parameter_


def _compile(src_dir: str, src_file: str, defines: Dict[str, int]) -> ctypes.CDLL:
    """Compile a prototype ``.cpp`` into a cached ``.so`` keyed by its defines, and
    load it. Mirrors each prototype's own ``compile_lib`` (one ``.so`` per shape)."""
    ascend = os.environ["ASCEND_HOME_PATH"]
    pto = os.environ["PTO_LIB_PATH"]
    arch = os.environ.get("NPU_ARCH", "dav-2201")
    inc = _einsum_include()
    dflags = [f"-D{k}={v}" for k, v in sorted(defines.items())]
    tag = hashlib.md5((src_file + "|" + " ".join(dflags) + "|" + arch).encode()).hexdigest()[:10]
    so = os.path.join(src_dir, f"fused_{os.path.splitext(src_file)[0]}_{tag}.so")
    if not os.path.exists(so):
        cmd = ["bisheng", "-O3", "-shared", "-fPIC", "-std=c++17", "-xcce",
               f"--npu-arch={arch}", *dflags,
               "-I", src_dir, "-I", inc, "-I", f"{ascend}/include", "-I", f"{pto}/include",
               "-L", f"{ascend}/lib64", "-lascendcl", "-lruntime",
               os.path.join(src_dir, src_file), "-o", so]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"fused kernel build failed ({src_file}):\n{r.stderr}")
    return ctypes.CDLL(so)


# --------------------------------------------------------------------------- #
#  contract + persistent runner
# --------------------------------------------------------------------------- #
@dataclass
class FusedContract:
    """The pinned interface of one fused kernel (documentation + the cast rule)."""
    in_slots: List[str]                  # logical input names, in launch order
    in_dtypes: List[Any]                 # required dtype per input slot
    out_names: List[str]                 # logical output names this kernel produces
    out_dtype: Any
    notes: str = ""


class FusedKernel:
    """A compiled fused ``.so`` bound to one shape, with its workspace set up once.

    ``run(inputs, params)`` adapts operands, launches the kernel on the current
    stream (no host sync — capturable), and returns the output tensors. The
    persistent workspace mirrors the library's persistent-runner discipline:
    setup/teardown are amortized, ``run`` is pure launch.
    """

    def __init__(self, lib: ctypes.CDLL, launch: Callable, setup_sym: str,
                 teardown_sym: str) -> None:
        self._lib = lib
        self._launch = launch
        self._teardown_sym = teardown_sym
        self._ws = ctypes.c_void_p(getattr(lib, setup_sym)())

    def run(self, inputs: List[torch.Tensor], params: dict) -> List[torch.Tensor]:
        return self._launch(self._lib, self._ws, inputs, params)

    def close(self) -> None:
        if self._ws:
            getattr(self._lib, self._teardown_sym)(self._ws)
            self._ws = ctypes.c_void_p(0)


@dataclass
class _Recipe:
    src_dir: str
    src_file: str
    defines: Callable[[dict], Dict[str, int]]   # params -> -D map (the .so key)
    setup_sym: str
    teardown_sym: str
    bind: Callable[[ctypes.CDLL], None]         # set argtypes/restype on the lib
    launch: Callable                            # (lib, ws, inputs, params) -> [out]
    contract: FusedContract


class FusedKernelRegistry:
    def __init__(self) -> None:
        self._recipes: Dict[str, _Recipe] = {}
        self._cache: Dict[Tuple, FusedKernel] = {}

    def register(self, key: str, recipe: _Recipe) -> None:
        if key in self._recipes:
            raise ValueError(f"fused kernel {key!r} already registered")
        self._recipes[key] = recipe

    def contract(self, key: str) -> FusedContract:
        return self._recipes[key].contract

    def kernel(self, key: str, params: dict) -> FusedKernel:
        """Build-or-fetch the persistent kernel for (key, shape params)."""
        if key not in self._recipes:
            raise KeyError(f"no fused kernel registered under {key!r}")
        r = self._recipes[key]
        defines = r.defines(params)
        ck = (key, tuple(sorted(defines.items())))
        if ck not in self._cache:
            lib = _compile(r.src_dir, r.src_file, defines)
            r.bind(lib)
            self._cache[ck] = FusedKernel(lib, r.launch, r.setup_sym, r.teardown_sym)
        return self._cache[ck]

    def run(self, key: str, inputs: List[torch.Tensor], params: dict) -> List[torch.Tensor]:
        return self.kernel(key, params).run(inputs, params)

    def close(self) -> None:
        for k in self._cache.values():
            k.close()
        self._cache.clear()


# --------------------------------------------------------------------------- #
#  chunk_h_scan — the resident-state recurrence
# --------------------------------------------------------------------------- #
def _scan_bind(lib: ctypes.CDLL) -> None:
    lib.scan_setup.restype = ctypes.c_void_p
    lib.scan_exec.argtypes = [ctypes.c_void_p] * 8
    lib.scan_teardown.argtypes = [ctypes.c_void_p]


def _scan_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    w, u, k, decay = inputs   # [B,nc,C,H,D] half ×3; decay [B,H,nc] f32 (scalar) or [B,H,nc,D] (per-dim)
    B, H, nc = params["B"], params["H"], params["nc"]
    dev = w.device
    h_out = torch.zeros(B, nc, H, SCAN_D, SCAN_D, device=dev, dtype=torch.float16)
    final = torch.zeros(B, H, SCAN_D, SCAN_D, device=dev, dtype=torch.float16)
    lib.scan_exec(_vp(w.contiguous()), _vp(u.contiguous()), _vp(k.contiguous()),
                  _vp(decay.contiguous().float()), ws, _vp(h_out), _vp(final), _stream())
    return [h_out, final]


# --------------------------------------------------------------------------- #
#  gated_qk — matmul-core (a@bᵀ) + on-chip gated/masked epilogue (glue absorption)
#  Shared by GDN's kkt (a=b=k, real beta, strict-lower mask) and chunk_o's Aqk
#  (a=q, b=k, beta=1, causal mask): one primitive covers both gated-matmul stages.
# --------------------------------------------------------------------------- #
def _kkt_bind(lib: ctypes.CDLL) -> None:
    _exec_argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int32, ctypes.c_int64, ctypes.c_void_p]
    lib.kkt_setup.restype = ctypes.c_void_p
    lib.kkt_exec.argtypes = _exec_argtypes
    lib.kkt_teardown.argtypes = [ctypes.c_void_p]
    # V2 (per-tile FFTS interleave): tiny L2-resident ring instead of a full I*C*C
    # ws_res — same ABI, distinct setup. Bound here so the v2 recipes can launch it.
    lib.kkt_setup_v2.restype = ctypes.c_void_p
    lib.kkt_exec_v2.argtypes = _exec_argtypes


def _gated_qk_run(lib, ws, a, b, g_sum, beta, nc, H, causal) -> List[torch.Tensor]:
    """Launch a@bᵀ + gated(exp(min(gᵢ-gⱼ+log βᵢ,0)))·mask epilogue -> L [T,H,C]."""
    T = nc * KKT_C
    dev = a.device
    a_flat = a.reshape(T, H, KKT_D).contiguous().half()
    b_flat = b.reshape(T, H, KKT_D).contiguous().half()
    g_t = g_sum[0].permute(1, 0).contiguous().float()      # [H,T]
    beta_t = beta[0].permute(1, 0).contiguous().half()     # [H,T]
    rows = torch.arange(KKT_C, device=dev)
    rel = rows[:, None] >= rows[None, :] if causal else rows[:, None] > rows[None, :]
    mask = rel.float().contiguous()                        # [C,C] causal / strict-lower
    L = torch.zeros(T, H, KKT_C, device=dev, dtype=torch.float16)
    lib.kkt_exec(_vp(a_flat), _vp(b_flat), _vp(g_t), _vp(beta_t), _vp(mask), ws, _vp(L),
                 ctypes.c_int32(H), ctypes.c_int64(T), _stream())
    return [L]


def _kkt_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    k, g_sum, beta = inputs                        # [1,T,H,D] half, [1,T,H] f32, [1,T,H] half
    return _gated_qk_run(lib, ws, k, k, g_sum, beta, params["nc"], params["H"], causal=False)


def _gated_qk_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    a, b, g_sum, beta = inputs                     # a@bᵀ; beta=1 + causal=True for chunk_o
    return _gated_qk_run(lib, ws, a, b, g_sum, beta, params["nc"], params["H"],
                         causal=params.get("causal", True))


# -- native [M,C,D] variants: operands feed straight in, no layout bridge -- #
def _gated_qk_run_native(lib, ws, a, b, g_native, beta_native, M, C, D, causal,
                         v2=False) -> List[torch.Tensor]:
    """Same gated a@bᵀ epilogue, but the kernel reads the Program's NATIVE [M,C,D]
    batch (heads outer) and writes L [M,C,C] — no transpose into the mega [1,T,H,D]
    layout, so q@kᵀ (chunk_o) pays zero operand shuffles. g/β are per-M-row [M,C].
    ``C``/``D`` are the chunk size and head dim (default 128 for GDN/KDA; the zoo runs
    smaller, e.g. C=16, D=64 — they key the .so via -DKKT_C/-DKKT_D, see ``defines``).
    ``v2`` selects the per-tile FFTS-interleave kernel (no qk HBM round-trip)."""
    dev = a.device
    a_flat = a.reshape(M, C, D).contiguous().half()
    b_flat = b.reshape(M, C, D).contiguous().half()
    g_t = g_native.reshape(M, C).contiguous().float()
    beta_t = beta_native.reshape(M, C).contiguous().half()
    rows = torch.arange(C, device=dev)
    rel = rows[:, None] >= rows[None, :] if causal else rows[:, None] > rows[None, :]
    mask = rel.float().contiguous()
    L = torch.zeros(M, C, C, device=dev, dtype=torch.float16)
    exec_fn = lib.kkt_exec_v2 if v2 else lib.kkt_exec
    exec_fn(_vp(a_flat), _vp(b_flat), _vp(g_t), _vp(beta_t), _vp(mask), ws, _vp(L),
            ctypes.c_int32(0), ctypes.c_int64(0), _stream())
    return [L]


def _kkt_native_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    k, g_native, beta_native = inputs              # [M,C,D] half, [M,C] f32, [M,C] half
    return _gated_qk_run_native(lib, ws, k, k, g_native, beta_native,
                                params["nc"] * params["H"],
                                params.get("C", KKT_C), params.get("D", KKT_D),
                                causal=False, v2=params.get("v2", False))


def _gated_qk_native_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    a, b, g_native, beta_native = inputs
    return _gated_qk_run_native(lib, ws, a, b, g_native, beta_native,
                                params["nc"] * params["H"],
                                params.get("C", KKT_C), params.get("D", KKT_D),
                                causal=params.get("causal", True), v2=params.get("v2", False))


def _kkt_native_v2_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    k, g_native, beta_native = inputs
    return _gated_qk_run_native(lib, ws, k, k, g_native, beta_native,
                                params["nc"] * params["H"],
                                params.get("C", KKT_C), params.get("D", KKT_D),
                                causal=False, v2=True)


def _gated_qk_native_v2_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    a, b, g_native, beta_native = inputs
    return _gated_qk_run_native(lib, ws, a, b, g_native, beta_native,
                                params["nc"] * params["H"],
                                params.get("C", KKT_C), params.get("D", KKT_D),
                                causal=params.get("causal", True), v2=True)


# --------------------------------------------------------------------------- #
#  qk_prologue — per-dim PROLOGUE: Vec q⊙P / k⊙invP -> matmul-core -> Vec mask.
#  The per-channel-gate (GLA/KDA) counterpart of gated_qk's scalar epilogue: the
#  decay rides on the OPERANDS (folded into the load) not a scalar coeff. V1 = three
#  passes in one launch (dispatch-elim; qd/kinv still round-trip a scratch).
# --------------------------------------------------------------------------- #
_pro_exec_argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int32, ctypes.c_int64, ctypes.c_void_p]


def _pro_bind(lib: ctypes.CDLL) -> None:
    lib.pro_setup.restype = ctypes.c_void_p
    lib.pro_exec.argtypes = _pro_exec_argtypes
    lib.pro_teardown.argtypes = [ctypes.c_void_p]
    # V2 (per-tile Vec->Cube->Vec, L2 ring): qd/kinv stay L2 — same ABI, distinct setup.
    lib.pro_setup_v2.restype = ctypes.c_void_p
    lib.pro_exec_v2.argtypes = _pro_exec_argtypes


def _pro_run(lib, ws, inputs, params, v2=False) -> List[torch.Tensor]:
    """A = tril((q⊙P) @ (k⊙invP)ᵀ) over all M=nc·H chunks. Inputs in Program order
    [q, k, P, invP], all [.,C,D] reshaping to [M,C,D]; the kernel keys C/D via -D.
    ``v2`` selects the L2-resident ring kernel (no full [M,C,D] qd/kinv scratch)."""
    q, k, P, invP = inputs
    C, D = params.get("C", KKT_C), params.get("D", KKT_D)
    M = params["nc"] * params["H"]
    dev = q.device
    rs = lambda t: t.reshape(M, C, D).contiguous().half()
    q_f, k_f, P_f, invP_f = rs(q), rs(k), rs(P), rs(invP)
    rows = torch.arange(C, device=dev)
    mask = (rows[:, None] >= rows[None, :]).float().contiguous()      # causal (incl diag)
    L = torch.zeros(M, C, C, device=dev, dtype=torch.float16)
    exec_fn = lib.pro_exec_v2 if v2 else lib.pro_exec
    exec_fn(_vp(q_f), _vp(P_f), _vp(k_f), _vp(invP_f), _vp(mask), ws, _vp(L),
            ctypes.c_int32(0), ctypes.c_int64(0), _stream())
    return [L]


def _pro_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    return _pro_run(lib, ws, inputs, params, v2=False)


def _pro_v2_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    return _pro_run(lib, ws, inputs, params, v2=True)


def _pro_defines(p: dict) -> Dict[str, int]:
    return {"PRO_NC": p["nc"], "PRO_H": p["H"],
            "PRO_C": p.get("C", KKT_C), "PRO_D": p.get("D", KKT_D)}


def _native_defines(p: dict) -> Dict[str, int]:
    """-D map keying a native-kkt .so: shape (nc,H) + chunk/head dims (C,D), default
    128 so GDN/KDA are unchanged; the zoo passes its own C/D to rebuild a distinct .so."""
    return {"KKT_NC": p["nc"], "KKT_H": p["H"], "KKT_NATIVE": 1,
            "KKT_C": p.get("C", KKT_C), "KKT_D": p.get("D", KKT_D)}


# --------------------------------------------------------------------------- #
#  qkv_flash — chunk_o score→output in ONE launch (B3). q·kᵀ → gate/mask → ·v,
#  so the [M,C,C] masked score never lands in HBM. Reuses the gated_qk epilogue
#  (kkt_epilogue_one) between two matmul-core passes; recomputes the exp gate +
#  causal mask on-chip from g_native, so it subsumes the coef_o mul AND the o_intra
#  A@v contraction the staged path runs as a separate einsum.
#    V1 (qkv_flash_native)    — two-pass (bulk Cube/Vec/Cube, S+A via HBM scratch).
#    V2 (qkv_flash_native_v2) — per-tile Cube-Vec-Cube ring, S+A L2-resident.
#  V1 measured faster at the shapes tested; V2 is bit-identical (the round-trip win
#  needs cross-tile overlap — a later tuning), so the template emits V1 by default.
# --------------------------------------------------------------------------- #
def _qkv_bind(lib: ctypes.CDLL) -> None:
    _exec_args = [ctypes.c_void_p] * 6 + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.qkv_setup.restype = ctypes.c_void_p
    lib.qkv_setup_v2.restype = ctypes.c_void_p
    lib.qkv_exec.argtypes = _exec_args
    lib.qkv_exec_v2.argtypes = _exec_args
    lib.qkv_teardown.argtypes = [ctypes.c_void_p]


def _qkv_defines(p: dict) -> Dict[str, int]:
    D = p.get("D", KKT_D)
    return {"QKV_NC": p["nc"], "QKV_H": p["H"], "QKV_C": p.get("C", KKT_C),
            "QKV_D": D, "QKV_DV": p.get("DV", D)}


def _qkv_run(lib, ws, inputs, params, v2=False) -> List[torch.Tensor]:
    """o = (tril((q·kᵀ)·exp(min(gᵢ-gⱼ,0)))) · v over all M=nc·H chunks, one launch.
    Inputs (Program order) [q, k, v, g_native, beta_native]; q/k [.,C,D], v [.,C,DV],
    g/beta per-row [.,C]. Output o [M,C,DV] fp32 (matches the o_intra einsum)."""
    q, k, v, g_native, beta_native = inputs
    C, D = params.get("C", KKT_C), params.get("D", KKT_D)
    DV = params.get("DV", D)
    M = params["nc"] * params["H"]
    dev = q.device
    q_f = q.reshape(M, C, D).contiguous().half()
    k_f = k.reshape(M, C, D).contiguous().half()
    v_f = v.reshape(M, C, DV).contiguous().half()
    g_t = g_native.reshape(M, C).contiguous().float()
    beta_t = beta_native.reshape(M, C).contiguous().half()
    rows = torch.arange(C, device=dev)
    mask = (rows[:, None] >= rows[None, :]).float().contiguous()      # causal incl diag
    o = torch.zeros(M, C, DV, device=dev, dtype=torch.float32)
    exec_fn = lib.qkv_exec_v2 if v2 else lib.qkv_exec
    exec_fn(_vp(q_f), _vp(k_f), _vp(v_f), _vp(g_t), _vp(beta_t), _vp(mask), ws, _vp(o), _stream())
    return [o]


def _qkv_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    return _qkv_run(lib, ws, inputs, params, v2=False)


def _qkv_v2_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    return _qkv_run(lib, ws, inputs, params, v2=True)


# --------------------------------------------------------------------------- #
#  qkvp_flash — PER-DIM chunk_o flash (KDA/GLA). The per-channel-gate twin of
#  qkv_flash: the decay rides on the OPERANDS (a Vec prescale q⊙coef_ag, k⊙coef_bg)
#  ahead of the score matmul, then a plain tril, then A·v — Vec→Cube→Vec→Cube in one
#  launch, ops+S+A all L2-resident. V2 is the double-buffered interleave (emitted); V1
#  is the four-pass bit-exact oracle.
# --------------------------------------------------------------------------- #
def _qkvp_bind(lib: ctypes.CDLL) -> None:
    _exec_args = [ctypes.c_void_p] * 6 + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.qkvp_setup.restype = ctypes.c_void_p
    lib.qkvp_setup_v2.restype = ctypes.c_void_p
    lib.qkvp_exec.argtypes = _exec_args
    lib.qkvp_exec_v2.argtypes = _exec_args
    lib.qkvp_teardown.argtypes = [ctypes.c_void_p]


def _qkvp_run(lib, ws, inputs, params, v2=False) -> List[torch.Tensor]:
    """o = tril((q⊙coef_ag)·(k⊙coef_bg)ᵀ) · v over all M=nc·H chunks, one launch.
    Inputs (Program order) [q, k, v, coef_ag, coef_bg]; q/k [.,C,D], v [.,C,DV],
    coef_ag/coef_bg [.,C,D] (per-dim exp(±g)). Output o [M,C,DV] fp32."""
    q, k, v, coef_ag, coef_bg = inputs
    C, D = params.get("C", KKT_C), params.get("D", KKT_D)
    DV = params.get("DV", D)
    M = params["nc"] * params["H"]
    dev = q.device
    q_f = q.reshape(M, C, D).contiguous().half()
    k_f = k.reshape(M, C, D).contiguous().half()
    v_f = v.reshape(M, C, DV).contiguous().half()
    ag = coef_ag.reshape(M, C, D).contiguous().half()
    bg = coef_bg.reshape(M, C, D).contiguous().half()
    rows = torch.arange(C, device=dev)
    mask = (rows[:, None] >= rows[None, :]).float().contiguous()      # causal incl diag
    o = torch.zeros(M, C, DV, device=dev, dtype=torch.float32)
    exec_fn = lib.qkvp_exec_v2 if v2 else lib.qkvp_exec
    exec_fn(_vp(q_f), _vp(ag), _vp(k_f), _vp(bg), _vp(mask), _vp(v_f), ws, _vp(o), _stream())
    return [o]


def _qkvp_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    return _qkvp_run(lib, ws, inputs, params, v2=False)


def _qkvp_v2_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    return _qkvp_run(lib, ws, inputs, params, v2=True)


def default_fused_registry() -> FusedKernelRegistry:
    reg = FusedKernelRegistry()
    reg.register("chunk_h_scan", _Recipe(
        src_dir=_KERNELS, src_file="scan_lib.cpp",
        defines=lambda p: {"SCAN_B": p["B"], "SCAN_H": p["H"], "SCAN_NC": p["nc"],
                           **({"SCAN_PERDIM_DECAY": 1} if p.get("perdim_decay") else {})},
        setup_sym="scan_setup", teardown_sym="scan_teardown",
        bind=_scan_bind, launch=_scan_launch,
        contract=FusedContract(
            in_slots=["w", "u", "k", "decay"],
            in_dtypes=[torch.float16, torch.float16, torch.float16, torch.float32],
            out_names=["h_out", "final"], out_dtype=torch.float16,
            notes="resident-state chunk recurrence; C=D=128; w/u/k [B,nc,C,H,D]; "
                  "decay [B,H,nc] scalar (GDN) or [B,H,nc,D] per-dim (KDA, "
                  "perdim_decay=True -> SCAN_PERDIM_DECAY keys a distinct .so)")))
    reg.register("kkt_gated", _Recipe(
        src_dir=_KERNELS, src_file="kkt_fused_lib.cpp",
        defines=lambda p: {"KKT_NC": p["nc"], "KKT_H": p["H"]},
        setup_sym="kkt_setup", teardown_sym="kkt_teardown",
        bind=_kkt_bind, launch=_kkt_launch,
        contract=FusedContract(
            in_slots=["k", "g_sum", "beta"],
            in_dtypes=[torch.float16, torch.float32, torch.float16],
            out_names=["L"], out_dtype=torch.float16,
            notes="matmul-core + on-chip gated+masked epilogue; C=D=128; k [1,T,H,D]")))
    reg.register("gated_qk", _Recipe(
        src_dir=_KERNELS, src_file="kkt_fused_lib.cpp",
        defines=lambda p: {"KKT_NC": p["nc"], "KKT_H": p["H"]},
        setup_sym="kkt_setup", teardown_sym="kkt_teardown",
        bind=_kkt_bind, launch=_gated_qk_launch,
        contract=FusedContract(
            in_slots=["a", "b", "g_sum", "beta"],
            in_dtypes=[torch.float16, torch.float16, torch.float32, torch.float16],
            out_names=["L"], out_dtype=torch.float16,
            notes="a@bᵀ matmul-core + on-chip gated+masked epilogue; same .so as "
                  "kkt_gated; chunk_o Aqk = (q,k,beta=1,causal); C=D=128; a,b [1,T,H,D]")))
    # native-[M,C,D] variants — same kernel, KKT_NATIVE config + epilogue path,
    # operands/g/β/L in the Program's own batch layout so there is no transpose bridge.
    reg.register("kkt_gated_native", _Recipe(
        src_dir=_KERNELS, src_file="kkt_fused_lib.cpp",
        defines=_native_defines,
        setup_sym="kkt_setup", teardown_sym="kkt_teardown",
        bind=_kkt_bind, launch=_kkt_native_launch,
        contract=FusedContract(
            in_slots=["k", "g_native", "beta_native"],
            in_dtypes=[torch.float16, torch.float32, torch.float16],
            out_names=["L"], out_dtype=torch.float16,
            notes="native [M,C,D] kkt (k@kᵀ, real beta, strict-lower); no layout "
                  "bridge; C,D via -DKKT_C/-DKKT_D (default 128); k [M,C,D], g/beta [M,C]")))
    reg.register("gated_qk_native", _Recipe(
        src_dir=_KERNELS, src_file="kkt_fused_lib.cpp",
        defines=_native_defines,
        setup_sym="kkt_setup", teardown_sym="kkt_teardown",
        bind=_kkt_bind, launch=_gated_qk_native_launch,
        contract=FusedContract(
            in_slots=["a", "b", "g_native", "beta_native"],
            in_dtypes=[torch.float16, torch.float16, torch.float32, torch.float16],
            out_names=["L"], out_dtype=torch.float16,
            notes="native [M,C,D] a@bᵀ + gated/masked epilogue; no layout bridge; "
                  "chunk_o Aqk = (q,k,beta=1,causal); C,D via -DKKT_C/-DKKT_D "
                  "(default 128); a,b [M,C,D], g/beta [M,C]")))
    # V2: native kernels with per-tile FFTS interleave (kkt_setup_v2 ring buffer) —
    # removes the kernel's own qk HBM round-trip, the residual vs the megakernel.
    reg.register("kkt_gated_native_v2", _Recipe(
        src_dir=_KERNELS, src_file="kkt_fused_lib.cpp",
        defines=_native_defines,
        setup_sym="kkt_setup_v2", teardown_sym="kkt_teardown",
        bind=_kkt_bind, launch=_kkt_native_v2_launch,
        contract=FusedContract(
            in_slots=["k", "g_native", "beta_native"],
            in_dtypes=[torch.float16, torch.float32, torch.float16],
            out_names=["L"], out_dtype=torch.float16,
            notes="native kkt + per-tile interleave (no qk round-trip); "
                  "C,D via -DKKT_C/-DKKT_D (default 128)")))
    reg.register("gated_qk_native_v2", _Recipe(
        src_dir=_KERNELS, src_file="kkt_fused_lib.cpp",
        defines=_native_defines,
        setup_sym="kkt_setup_v2", teardown_sym="kkt_teardown",
        bind=_kkt_bind, launch=_gated_qk_native_v2_launch,
        contract=FusedContract(
            in_slots=["a", "b", "g_native", "beta_native"],
            in_dtypes=[torch.float16, torch.float16, torch.float32, torch.float16],
            out_names=["L"], out_dtype=torch.float16,
            notes="native a@bᵀ + per-tile interleave (no qk round-trip); "
                  "C,D via -DKKT_C/-DKKT_D (default 128)")))
    # Per-dim PROLOGUE (GLA/KDA): operand prescale q⊙P / k⊙invP folded into one launch
    # ahead of the matmul-core, then a causal/tril mask. The per-channel-gate twin of
    # gated_qk's scalar epilogue. V1: dispatch-elim (qd/kinv round-trip a scratch).
    reg.register("qk_prologue", _Recipe(
        src_dir=_KERNELS, src_file="prologue_fused_lib.cpp",
        defines=_pro_defines,
        setup_sym="pro_setup", teardown_sym="pro_teardown",
        bind=_pro_bind, launch=_pro_launch,
        contract=FusedContract(
            in_slots=["q", "k", "P", "invP"],
            in_dtypes=[torch.float16, torch.float16, torch.float16, torch.float16],
            out_names=["L"], out_dtype=torch.float16,
            notes="per-dim prologue: tril((q⊙P)@(k⊙invP)ᵀ); per-channel gate (GLA/KDA); "
                  "C,D via -DPRO_C/-DPRO_D; q/k/P/invP [M,C,D], L [M,C,C]")))
    reg.register("qk_prologue_v2", _Recipe(
        src_dir=_KERNELS, src_file="prologue_fused_lib.cpp",
        defines=_pro_defines,
        setup_sym="pro_setup_v2", teardown_sym="pro_teardown",
        bind=_pro_bind, launch=_pro_v2_launch,
        contract=FusedContract(
            in_slots=["q", "k", "P", "invP"],
            in_dtypes=[torch.float16, torch.float16, torch.float16, torch.float16],
            out_names=["L"], out_dtype=torch.float16,
            notes="per-dim prologue, L2-resident ring (Vec prescale->Cube matmul from "
                  "slot->Vec mask); qd/kinv never hit HBM; C,D via -DPRO_C/-DPRO_D")))
    # qkv_flash: chunk_o score→output fused (q·kᵀ → gate/mask → ·v), score never in HBM.
    _qkv_contract = FusedContract(
        in_slots=["q", "k", "v", "g_native", "beta_native"],
        in_dtypes=[torch.float16, torch.float16, torch.float16, torch.float32, torch.float16],
        out_names=["o"], out_dtype=torch.float32,
        notes="chunk_o flash: o = tril((q·kᵀ)·exp(min(gᵢ-gⱼ,0)))·v in ONE launch; "
              "recomputes the gate+causal mask on-chip (subsumes coef_o mul + o_intra "
              "A@v); C,D,DV via -DQKV_C/_D/_DV; q/k [M,C,D], v [M,C,DV], o [M,C,DV] f32")
    reg.register("qkv_flash_native", _Recipe(
        src_dir=_KERNELS, src_file="qkv_fused_lib.cpp", defines=_qkv_defines,
        setup_sym="qkv_setup", teardown_sym="qkv_teardown",
        bind=_qkv_bind, launch=_qkv_launch, contract=_qkv_contract))
    reg.register("qkv_flash_native_v2", _Recipe(
        src_dir=_KERNELS, src_file="qkv_fused_lib.cpp", defines=_qkv_defines,
        setup_sym="qkv_setup_v2", teardown_sym="qkv_teardown",
        bind=_qkv_bind, launch=_qkv_v2_launch, contract=_qkv_contract))
    # qkvp_flash: per-dim (KDA/GLA) chunk_o flash — operand prescale + score + tril + A·v.
    _qkvp_contract = FusedContract(
        in_slots=["q", "k", "v", "coef_ag", "coef_bg"],
        in_dtypes=[torch.float16, torch.float16, torch.float16, torch.float16, torch.float16],
        out_names=["o"], out_dtype=torch.float32,
        notes="per-dim chunk_o flash: o = tril((q⊙coef_ag)·(k⊙coef_bg)ᵀ)·v in ONE launch "
              "(Vec prescale→Cube score→Vec tril→Cube A·v); ops+S+A L2-resident; "
              "C,D,DV via -DQKV_C/_D/_DV; q/k/coef [M,C,D], v [M,C,DV], o [M,C,DV] f32")
    reg.register("qkvp_flash_native", _Recipe(
        src_dir=_KERNELS, src_file="qkv_prologue_fused_lib.cpp", defines=_qkv_defines,
        setup_sym="qkvp_setup", teardown_sym="qkvp_teardown",
        bind=_qkvp_bind, launch=_qkvp_launch, contract=_qkvp_contract))
    reg.register("qkvp_flash_native_v2", _Recipe(
        src_dir=_KERNELS, src_file="qkv_prologue_fused_lib.cpp", defines=_qkv_defines,
        setup_sym="qkvp_setup_v2", teardown_sym="qkvp_teardown",
        bind=_qkvp_bind, launch=_qkvp_v2_launch, contract=_qkvp_contract))
    return reg


_DEFAULT: Optional[FusedKernelRegistry] = None


def shared_fused_registry() -> FusedKernelRegistry:
    """A process-wide registry so persistent kernels are reused across executors
    (and survive across graph capture/replay)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = default_fused_registry()
    return _DEFAULT
