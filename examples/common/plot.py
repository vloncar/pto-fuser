"""Plotting helpers (matplotlib, optional).

Kept dependency-light and import-guarded: if matplotlib is not installed the helpers
return ``None`` and the caller still gets its table. Mirrors the plotting style of
the pto-einsum ``benchmarks/complex/*/utils.py`` (a saved ``.png`` next to the data).
"""
from __future__ import annotations

from typing import List, Optional, Sequence


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_speedups(labels: Sequence[str], speedups: Sequence[float], path: str, *,
                  title: str = "", baseline_label: str = "baseline (1.0×)") -> Optional[str]:
    """Horizontal bar chart of per-variant speedups vs a baseline, saved to ``path``.

    Returns ``path`` on success, ``None`` if matplotlib is unavailable.
    """
    plt = _mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(labels) + 1.5))
    ax.barh(range(len(labels)), speedups, color="#4C72B0")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(1.0, color="#888", linestyle="--", linewidth=1, label=baseline_label)
    for i, s in enumerate(speedups):
        ax.text(s, i, f" {s:.2f}×", va="center", fontsize=9)
    ax.set_xlabel("speedup vs baseline (higher is better)")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_sweep(x: Sequence[float], series: dict, path: str, *, title: str = "",
               xlabel: str = "", ylabel: str = "ms/call", logx: bool = False) -> Optional[str]:
    """Line plot of one or more ``{label: y-values}`` series over ``x``, saved to
    ``path``. Returns ``path`` on success, ``None`` if matplotlib is unavailable."""
    plt = _mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, ys in series.items():
        ax.plot(x, ys, marker="o", label=label)
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
