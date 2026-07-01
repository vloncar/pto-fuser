"""Transform-layer unit tests (no NPU) — the pure IR rewrites in isolation.

Covers canonicalization, the universal read-mode / fused-store annotation levers, and
the structural fusion transforms: that each matches only its own forward family,
inserts the expected FusedNode, is idempotent, and leaves a valid Program.
"""
import pytest

from pto_fuser import (EinsumNode, FusedNode, canonicalize, EnableDirectReads,
                       EnableFusedStore, FuseContractionEpilogue)
from pto_fuser.transforms import LowerPerDimScan, LowerResidentScan


def _gdn():
    from attention._gdn_full import build_gdn_program
    return build_gdn_program(1, 4, 2, 128, 128, 128 ** -0.5)


def _kda():
    from attention._kda_full import build_kda_program
    return build_kda_program(1, 4, 2, 128, 128, 128 ** -0.5)


# --------------------------------------------------------------------------- #
#  canonical form + annotation levers
# --------------------------------------------------------------------------- #
def test_canonicalize_forces_nn():
    canon = canonicalize(_gdn())
    einsums = [n for n in canon.nodes if isinstance(n, EinsumNode)]
    assert einsums and all(n.read_mode == "NN" and not n.fuse_out for n in einsums)


def test_enable_direct_reads_is_idempotent():
    canon = canonicalize(_gdn())
    t = EnableDirectReads()
    n = t.match(canon)
    assert n == sum(isinstance(x, EinsumNode) for x in canon.nodes) > 0
    res = t.apply(canon)
    assert res.changed and res.sites == n
    assert all(x.read_mode == "auto" for x in res.program.nodes
               if isinstance(x, EinsumNode))
    assert t.match(res.program) == 0            # nothing left to promote


def test_enable_fused_store_is_idempotent():
    canon = canonicalize(_gdn())
    t = EnableFusedStore()
    res = t.apply(canon)
    assert res.changed
    assert all(x.fuse_out for x in res.program.nodes if isinstance(x, EinsumNode))
    assert t.match(res.program) == 0


# --------------------------------------------------------------------------- #
#  structural transforms match only their own family + are idempotent
# --------------------------------------------------------------------------- #
def test_resident_scan_matches_gdn_not_kda():
    assert LowerResidentScan(1, 4, 2, 128, 128).match(_gdn()) == 1
    assert LowerResidentScan(1, 4, 2, 128, 128).match(_kda()) == 0   # KDA is per-dim


def test_perdim_scan_matches_kda_not_gdn():
    assert LowerPerDimScan(1, 4, 2, 128, 128).match(_kda()) == 1
    assert LowerPerDimScan(1, 4, 2, 128, 128).match(_gdn()) == 0


def test_generator_matches_both_forwards():
    # GDN: kkt (k·kᵀ) + chunk_o (q·kᵀ) both match proven templates
    assert FuseContractionEpilogue(1, 4, 2, 128, 128).match(_gdn()) == 2
    # KDA: only chunk_o (per-dim prologue) matches; kkt stays a plain einsum
    assert FuseContractionEpilogue(1, 4, 2, 128, 128).match(_kda()) == 1


def test_generator_emits_proven_kernels():
    gdn = FuseContractionEpilogue(1, 4, 2, 128, 128).apply(canonicalize(_gdn())).program
    assert {n.kernel for n in gdn.nodes if isinstance(n, FusedNode)} \
        == {"kkt_gated_native", "gated_qk_native"}
    kda = FuseContractionEpilogue(1, 4, 2, 128, 128).apply(canonicalize(_kda())).program
    kk = {n.kernel for n in kda.nodes if isinstance(n, FusedNode)}
    assert len(kk) == 1 and next(iter(kk)).startswith("qk_prologue")   # shape-gates v1/v2


def test_resident_scan_idempotent_and_valid():
    canon = canonicalize(_gdn())
    t = LowerResidentScan(1, 4, 2, 128, 128)
    prog = t.apply(canon).program                # Program() re-validates on construction
    assert any(isinstance(n, FusedNode) and n.kernel == "chunk_h_scan"
               for n in prog.nodes)
    assert t.match(prog) == 0                     # already fused -> no re-match
    assert "S0" not in {getattr(n, "output", None) for n in prog.nodes}


def test_transforms_compose_into_valid_program():
    canon = canonicalize(_gdn())
    prog = LowerResidentScan(1, 4, 2, 128, 128).apply(canon).program
    prog = FuseContractionEpilogue(1, 4, 2, 128, 128).apply(prog).program
    kernels = {n.kernel for n in prog.nodes if isinstance(n, FusedNode)}
    assert kernels == {"chunk_h_scan", "kkt_gated_native", "gated_qk_native"}
    assert prog.outputs == ["o"]                  # still produces the forward output
