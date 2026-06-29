"""Fused-kernel registry — the lever-5/6 backend (design §4, §5).

A ``FusedNode`` names a hand-fused kernel that subsumes a staged sub-chain into one
dispatch, keeping intermediates on-chip. The registry maps that name to a recipe
that (lazily) compiles the prototype ``.cpp`` into a cached ``.so`` and to a
persistent runner: ``setup`` allocates the on-chip-state workspace once, ``run``
launches the kernel (no host sync — so the node is graph-capturable, M3), and the
runner is reused across calls. Operand layout/dtype adapters and output allocation
live in the registered lowering, exactly as the opaque registry does for tri_inv.

Two seed entries, both proven prototypes built as their own ``.so`` sharing GM with
the surrounding stages (design §9: *this* form works today; single-``.so`` opaque
inline is the unproven part and is not used here):

  * ``chunk_h_scan`` — the T2 resident-state recurrence (lever 5). State ``S`` stays
    resident across chunks (matmul operand + Vec accumulator); the staged lowering
    round-trips ``S`` through HBM every chunk.
  * ``kkt_gated``    — the T0 matmul-core + on-chip gated/masked epilogue (lever 6 /
    glue absorption). The qk intermediate never lands in HBM.

Compile-time shape (``KKT_NC``/``KKT_H``; ``SCAN_B``/``SCAN_H``/``SCAN_NC``) keys the
``.so``; ``C = D = 128`` are fixed in the prototypes.
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
_PROTO = os.path.abspath(os.path.join(HERE, "..", "..", "prototypes"))
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
    persistent workspace mirrors the substrate's persistent-runner discipline (M3):
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
#  chunk_h_scan — the T2 resident-state recurrence (lever 5)
# --------------------------------------------------------------------------- #
def _scan_bind(lib: ctypes.CDLL) -> None:
    lib.scan_setup.restype = ctypes.c_void_p
    lib.scan_exec.argtypes = [ctypes.c_void_p] * 8
    lib.scan_teardown.argtypes = [ctypes.c_void_p]


def _scan_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    w, u, k, decay = inputs                       # [B,nc,C,H,D] half ×3, [B,H,nc] f32
    B, H, nc = params["B"], params["H"], params["nc"]
    dev = w.device
    h_out = torch.zeros(B, nc, H, SCAN_D, SCAN_D, device=dev, dtype=torch.float16)
    final = torch.zeros(B, H, SCAN_D, SCAN_D, device=dev, dtype=torch.float16)
    lib.scan_exec(_vp(w.contiguous()), _vp(u.contiguous()), _vp(k.contiguous()),
                  _vp(decay.contiguous().float()), ws, _vp(h_out), _vp(final), _stream())
    return [h_out, final]


# --------------------------------------------------------------------------- #
#  kkt_gated — matmul-core + on-chip gated/masked epilogue (lever 6)
# --------------------------------------------------------------------------- #
def _kkt_bind(lib: ctypes.CDLL) -> None:
    lib.kkt_setup.restype = ctypes.c_void_p
    lib.kkt_exec.argtypes = [ctypes.c_void_p] * 4 + [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64, ctypes.c_void_p]
    lib.kkt_teardown.argtypes = [ctypes.c_void_p]


def _kkt_launch(lib, ws, inputs, params) -> List[torch.Tensor]:
    k, g_sum, beta = inputs                        # [1,T,H,D] half, [1,T,H] f32, [1,T,H] half
    nc, H = params["nc"], params["H"]
    T = nc * KKT_C
    dev = k.device
    k_flat = k.reshape(T, H, KKT_D).contiguous().half()
    g_t = g_sum[0].permute(1, 0).contiguous().float()      # [H,T]
    beta_t = beta[0].permute(1, 0).contiguous().half()     # [H,T]
    rows = torch.arange(KKT_C, device=dev)
    mask = (rows[:, None] > rows[None, :]).float().contiguous()   # [C,C] strict-lower
    L = torch.zeros(T, H, KKT_C, device=dev, dtype=torch.float16)
    lib.kkt_exec(_vp(k_flat), _vp(g_t), _vp(beta_t), _vp(mask), ws, _vp(L),
                 ctypes.c_int32(H), ctypes.c_int64(T), _stream())
    return [L]


def default_fused_registry() -> FusedKernelRegistry:
    reg = FusedKernelRegistry()
    reg.register("chunk_h_scan", _Recipe(
        src_dir=os.path.join(_PROTO, "chunk_h_scan"), src_file="scan_lib.cpp",
        defines=lambda p: {"SCAN_B": p["B"], "SCAN_H": p["H"], "SCAN_NC": p["nc"]},
        setup_sym="scan_setup", teardown_sym="scan_teardown",
        bind=_scan_bind, launch=_scan_launch,
        contract=FusedContract(
            in_slots=["w", "u", "k", "decay"],
            in_dtypes=[torch.float16, torch.float16, torch.float16, torch.float32],
            out_names=["h_out", "final"], out_dtype=torch.float16,
            notes="resident-state chunk recurrence; C=D=128; w/u/k [B,nc,C,H,D]")))
    reg.register("kkt_gated", _Recipe(
        src_dir=os.path.join(_PROTO, "kkt_fused"), src_file="kkt_fused_lib.cpp",
        defines=lambda p: {"KKT_NC": p["nc"], "KKT_H": p["H"]},
        setup_sym="kkt_setup", teardown_sym="kkt_teardown",
        bind=_kkt_bind, launch=_kkt_launch,
        contract=FusedContract(
            in_slots=["k", "g_sum", "beta"],
            in_dtypes=[torch.float16, torch.float32, torch.float16],
            out_names=["L"], out_dtype=torch.float16,
            notes="matmul-core + on-chip gated+masked epilogue; C=D=128; k [1,T,H,D]")))
    return reg


_DEFAULT: Optional[FusedKernelRegistry] = None


def shared_fused_registry() -> FusedKernelRegistry:
    """A process-wide registry so persistent kernels are reused across executors
    (and survive across graph capture/replay)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = default_fused_registry()
    return _DEFAULT
