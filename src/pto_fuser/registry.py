"""Opaque-kernel registry.

An ``OpaqueNode`` names a hand-optimized kernel the matmul-core cannot express.
The registry maps that name to a factory that (lazily) builds the callable and to
the pinned contract — required dtypes, adapters, the dtype cast. Keeping the
contract here, not in the graph, is what makes "dtype is part of the contract"
enforceable: the lowering passes a tensor through the registered adapter, which
casts to the kernel's required dtype before launch.

The seed entry is ``tri_inv_rec_unroll`` — the DeltaNet triangular inverse, reused
from pto-kernels' own JIT build (rec_unroll). Its contract, pinned by
`prototypes/deltanet_chunk/probe_triinv.py`:

  * inverts strictly-UPPER unit-triangular matrices, so a strictly-LOWER ``A`` is
    fed transposed and the result transposed back;
  * fp16 input, fp32 output;
  * ``matrix_size`` (chunk side C) in {16, 32, 64, 128};
  * the raw ctypes launch needs a sync on each side to order against the torch
    stream.
"""
from __future__ import annotations

import os
import sys
import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch


# When True, opaque lowerings omit their eager stream-ordering `synchronize()`s.
# The graph-replay backend (graph.py) sets this *inside the capture region only*:
# there the raw kernel launch is recorded on the capture stream in program order,
# so the host syncs are both unnecessary and capture-breaking (a host sync aborts
# capture). Outside capture the syncs are load-bearing — leave this False.
_CAPTURE_MODE = False


@contextlib.contextmanager
def capture_mode():
    """Mark a region as graph-capture: opaque lowerings drop their host syncs."""
    global _CAPTURE_MODE
    prev = _CAPTURE_MODE
    _CAPTURE_MODE = True
    try:
        yield
    finally:
        _CAPTURE_MODE = prev


@dataclass
class OpaqueContract:
    """The pinned interface of one opaque kernel (documentation + the cast rule)."""
    in_dtypes: List[Any]                 # required dtype per input slot
    out_dtype: Any
    notes: str = ""


@dataclass
class _Entry:
    factory: Callable[[], Callable]      # builds the raw callable (compiled once)
    lowering: Callable                   # (raw, inputs, params) -> output tensor
    contract: OpaqueContract
    _raw: Optional[Callable] = field(default=None, repr=False)

    def callable(self) -> Callable:
        if self._raw is None:
            self._raw = self.factory()
        return self._raw


class OpaqueRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, _Entry] = {}

    def register(self, key: str, *, factory: Callable, lowering: Callable,
                 contract: OpaqueContract) -> None:
        if key in self._entries:
            raise ValueError(f"opaque kernel {key!r} already registered")
        self._entries[key] = _Entry(factory, lowering, contract)

    def contract(self, key: str) -> OpaqueContract:
        return self._entries[key].contract

    def run(self, key: str, inputs: List[torch.Tensor], params: dict) -> torch.Tensor:
        if key not in self._entries:
            raise KeyError(f"no opaque kernel registered under {key!r}")
        entry = self._entries[key]
        return entry.lowering(entry.callable(), inputs, params)


# --------------------------------------------------------------------------- #
#  tri_inv_rec_unroll — the DeltaNet triangular inverse (pto-kernels rec_unroll)
# --------------------------------------------------------------------------- #
def _tri_inv_factory() -> Callable:
    """Compile the pto-kernels fast_inverse kernel and return its callable."""
    pto_kernels = os.environ.get(
        "PTO_KERNELS", "/home/vloncar/work/einsum_workspace/pto-kernels")
    fast_inv_dir = os.path.join(pto_kernels, "examples", "jit_cpp", "fast_inverse")
    if fast_inv_dir not in sys.path:
        sys.path.insert(0, fast_inv_dir)
    from jit_util_fast_inverse import jit_compile  # noqa: import-after-path
    return jit_compile(os.path.join(fast_inv_dir, "fast_inverse.cpp"), verbose=False)


def _tri_inv_lowering(tri_inv: Callable, inputs: List[torch.Tensor],
                      params: dict) -> torch.Tensor:
    """Invert (I + A_lower) per matrix via the opaque rec_unroll kernel.

    Contract (probe_triinv.py): the kernel inverts strictly-UPPER unit matrices, so
    feed A^T (strictly upper) and transpose the fp32 result back. Dtype is part of
    the contract — A is cast to fp16 here regardless of its incoming dtype.
    """
    (A_lower,) = inputs
    A_lower = A_lower.half()                       # the cast the kernel requires
    M, C, _ = A_lower.shape
    A_upper = A_lower.transpose(-1, -2).contiguous()
    out = torch.zeros_like(A_upper, dtype=torch.float32)
    mi = torch.zeros((C, C), dtype=torch.float16, device=A_lower.device)
    mi.fill_diagonal_(-1)
    if not _CAPTURE_MODE:
        torch.npu.synchronize()                    # order the raw launch vs the stream
    tri_inv(out, A_upper, mi, C, M, 0, cu_seqlens=None, block_dim=min(20, M))
    if not _CAPTURE_MODE:                           # under capture the launch is
        torch.npu.synchronize()                    # recorded on the capture stream
    return out.transpose(-1, -2).contiguous()      # (I+A)^-1, lower, fp32


def default_registry() -> OpaqueRegistry:
    reg = OpaqueRegistry()
    reg.register(
        "tri_inv_rec_unroll",
        factory=_tri_inv_factory,
        lowering=_tri_inv_lowering,
        contract=OpaqueContract(
            in_dtypes=[torch.float16], out_dtype=torch.float32,
            notes="strictly-lower unit-triangular A -> (I+A)^-1; C in {16,32,64,128}"),
    )
    return reg
