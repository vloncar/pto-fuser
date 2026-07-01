"""Head-to-head stages for the staged-vs-fused decision (see DESIGN.md).

Each stage is provided in **two** equivalent lowerings over identical inputs, so
`fusion.decide` can gate (frob ≡ + determinism) and time them against each other:

  * a **staged** `Program` — the default lowering the planner would emit (einsum
    cores + Vec glue), run through the staged/captured backend;
  * a **fused** `Program` — a single `FusedNode` hosting the proven prototype kernel
    that keeps the stage's intermediate on-chip.

Two stages, the two fusion features:

  * ``chunk_h_scan`` (resident state): the staged lowering unrolls the
    cross-chunk recurrence into ``nc`` per-chunk matmul pairs, round-tripping the
    state ``S`` through HBM every chunk; the fused kernel keeps ``S`` resident.
  * ``kkt_gated`` (glue absorption): the staged lowering is the qk
    contraction + a gated/masked Vec epilogue with the qk matrix landing in HBM; the
    fused kernel folds the epilogue into the matmul store.

``C = D = 128`` throughout (the prototype kernels' fixed tile).
"""
from __future__ import annotations

import torch

from ..ir import EinsumNode, FusedNode, Program, TensorOp, VecGlueNode

C = D = 128


# --------------------------------------------------------------------------- #
#  chunk_h_scan  (resident state)
# --------------------------------------------------------------------------- #
def make_scan_inputs(B: int, H: int, nc: int, device="npu:0") -> dict:
    """Prototype-layout operands (token-major [B,nc,C,H,D]); contractive scale so
    the fp16 recurrence stays bounded over nc chunks (synthetic-stability only —
    mirrors prototypes/chunk_h_scan/run_scan.py)."""
    g = dict(device=device)
    w = (torch.randn(B, nc, C, H, D, **g) * 0.02).half()
    k = (torch.randn(B, nc, C, H, D, **g) * 0.02).half()
    u = (torch.randn(B, nc, C, H, D, **g) * 0.05).half()
    decay = (0.7 + 0.2 * torch.rand(B, H, nc, **g)).float()
    return {"w": w, "u": u, "k": k, "decay": decay}


def scan_reference(inp: dict, B: int, H: int, nc: int) -> dict:
    """Clean fp32 reference of the simplified chunk_h recurrence (run_scan.py)."""
    w, u, k, decay = (inp[n] for n in ("w", "u", "k", "decay"))
    dev = w.device
    h_out = torch.zeros(B, nc, H, D, D, device=dev)
    final = torch.zeros(B, H, D, D, device=dev)
    wf, uf, kf, df = w.float(), u.float(), k.float(), decay.float()
    for b in range(B):
        for h in range(H):
            S = torch.zeros(D, D, device=dev)
            for c in range(nc):
                h_out[b, c, h] = S
                WS = wf[b, c, :, h, :] @ S
                vc = uf[b, c, :, h, :] - WS
                S = df[b, h, c] * S + kf[b, c, :, h, :].T @ vc
            final[b, h] = S
    return {"h_out": h_out, "final": final}


def build_scan_fused_program(B: int, H: int, nc: int) -> Program:
    """One `FusedNode` hosting the resident-state kernel."""
    node = FusedNode(kernel="chunk_h_scan",
                     inputs=["w", "u", "k", "decay"],
                     outputs=["h_out", "final"],
                     params={"B": B, "H": H, "nc": nc},
                     subsumes=["the nc per-chunk WS/kv matmul pairs + residual glue"])
    return Program(nodes=[node], inputs=["w", "u", "k", "decay"],
                   outputs=["h_out", "final"])


def build_scan_staged_program(B: int, H: int, nc: int) -> Program:
    """The per-chunk unrolled scan as einsum cores + Vec glue (the staged lowering
    graph capture replays). State ``S`` round-trips HBM each chunk — exactly what the
    fused kernel removes. Operands are batched over ``N = B*H`` (the einsum leading
    batch); outputs are reshaped back to the canonical [B,nc,H,D,D] / [B,H,D,D]."""
    N = B * H
    fp16 = torch.float16
    nodes: list = []

    # batch the token-major inputs over (b,h): [B,nc,C,H,D] -> [N,nc,C,D]
    for name in ("w", "u", "k"):
        nodes.append(TensorOp("permute", [name], f"{name}_p", {"dims": (0, 3, 1, 2, 4)}))
        nodes.append(TensorOp("reshape", [f"{name}_p"], f"{name}_b",
                              {"shape": (N, nc, C, D)}))
    nodes.append(TensorOp("reshape", ["decay"], "decay_b", {"shape": (N, nc)}))
    nodes.append(TensorOp("zeros", ["w_b"], "S0",
                          {"shape": (N, D, D), "dtype": fp16}))

    readouts = []
    S = "S0"
    for c in range(nc):
        readouts.append(S)                                  # h_out[c] = S (pre-update)
        wc, kc = f"w_c{c}", f"k_c{c}"
        nodes.append(TensorOp("slice", ["w_b"], wc, {"axis": 1, "index": c}))
        nodes.append(TensorOp("slice", ["k_b"], kc, {"axis": 1, "index": c}))
        nodes.append(TensorOp("slice", ["u_b"], f"u_c{c}", {"axis": 1, "index": c}))
        # WS = w_c @ S   (contraction d)
        nodes.append(EinsumNode("ncd,nde->nce", [wc, S], f"WS{c}"))
        # vc = u_c - WS  (Vec residual; cast back to fp16 for the next matmul)
        nodes.append(VecGlueNode("sub", [f"u_c{c}", f"WS{c}"], f"vc{c}", out_dtype=fp16))
        # kv = k_c^T @ vc  (contraction over the chunk tokens c)
        nodes.append(EinsumNode("ncd,nce->nde", [kc, f"vc{c}"], f"kv{c}"))
        # S' = decay_c * S + kv
        nodes.append(TensorOp("slice", ["decay_b"], f"dec{c}", {"axis": 1, "index": c}))
        nodes.append(TensorOp("reshape", [f"dec{c}"], f"decr{c}", {"shape": (N, 1, 1)}))
        nodes.append(VecGlueNode("mul", [S, f"decr{c}"], f"Sdec{c}"))
        S = f"S{c + 1}"
        nodes.append(VecGlueNode("add", [f"Sdec{c}", f"kv{c}"], S, out_dtype=fp16))

    # canonical outputs: [N,nc,D,D] -> [B,nc,H,D,D];  final [N,D,D] -> [B,H,D,D]
    nodes.append(TensorOp("stack", readouts, "h_b", {"dim": 1}))
    nodes.append(TensorOp("reshape", ["h_b"], "h_bhncdd", {"shape": (B, H, nc, D, D)}))
    nodes.append(TensorOp("permute", ["h_bhncdd"], "h_out", {"dims": (0, 2, 1, 3, 4)}))
    nodes.append(TensorOp("reshape", [S], "final", {"shape": (B, H, D, D)}))
    return Program(nodes=nodes, inputs=["w", "u", "k", "decay"],
                   outputs=["h_out", "final"])


# --------------------------------------------------------------------------- #
#  kkt_gated  (glue absorption)
# --------------------------------------------------------------------------- #
def make_kkt_inputs(nc: int, H: int, device="npu:0") -> dict:
    """Operands for the gated kkt: normalized k [1,T,H,D] half, prefix-summed gate
    g_sum [1,T,H] f32, and beta [1,T,H] half (mirrors run_kkt.py with Hg = H)."""
    T = nc * C
    k = torch.nn.functional.normalize(torch.randn(1, T, H, D, device=device), dim=-1).half()
    beta = torch.rand(1, T, H, device=device).half()
    g_in = torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=device).float())
    g_sum = torch.zeros_like(g_in)
    for j in range(0, T, C):
        g_sum[:, j:j + C] = g_in[:, j:j + C].cumsum(1)
    return {"k": k, "g_sum": g_sum, "beta": beta}


def kkt_reference(inp: dict, nc: int, H: int) -> dict:
    """fp32 reference of the gated/masked kkt the fused kernel computes (run_kkt.py):
    L = (qk · coeff · strict-lower-mask), qk = k k^T per (chunk, head)."""
    k, g_sum, beta = (inp[n] for n in ("k", "g_sum", "beta"))
    dev = k.device
    T = nc * C
    kcr = k.reshape(nc, C, H, D).float()
    qk = torch.einsum("cihd,cjhd->cihj", kcr, kcr)
    gs = g_sum[0].reshape(nc, C, H)
    bs = beta[0].reshape(nc, C, H).float()
    gv = gs + torch.log(bs)
    diff = gv.permute(0, 2, 1).unsqueeze(-1) - gs.permute(0, 2, 1).unsqueeze(2)   # [nc,H,i,j]
    coeff = torch.exp(torch.clamp(diff, max=0.0)).permute(0, 2, 1, 3)             # [nc,i,H,j]
    rows = torch.arange(C, device=dev)
    m = (rows[:, None] > rows[None, :]).float()
    L = (qk * coeff * m[None, :, None, :]).reshape(T, H, C).half()
    return {"L": L}


def build_kkt_fused_program(nc: int, H: int) -> Program:
    node = FusedNode(kernel="kkt_gated", inputs=["k", "g_sum", "beta"],
                     outputs=["L"], params={"nc": nc, "H": H},
                     subsumes=["qk einsum + gated/masked Vec epilogue"])
    return Program(nodes=[node], inputs=["k", "g_sum", "beta"], outputs=["L"])
