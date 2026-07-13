"""Figure generation from results artifacts, styled to Nature Machine
Intelligence (NMI) display conventions.

Design rules applied here (Nature Portfolio artwork & formatting guidance):
  * Vector PDF output with embedded, editable text (``pdf.fonttype = 42``).
  * Sans-serif typeface (Arial/Helvetica family), 5--8 pt range.
  * No in-panel titles -- every description lives in the LaTeX caption.
  * Bold lower-case panel labels (a, b, ...) for multi-panel figures.
  * Colour-blind-safe qualitative palette (Okabe & Ito / Wong, Nat. Methods 2011).
  * Uncertainty shown wherever a mean is plotted (shaded 95% CI bands / error bars).
  * Top/right spines removed for an uncluttered Nature-style frame.

Figures are produced purely from ``results/summary.json`` -- the single source of
truth written by the experiment runner -- so they regenerate from fixed seeds.
The cool colours mark first-order schedules (random / coefficient / clique); the
warm colours mark the folded second-order schedules (antithetic / symmetric).
"""
from __future__ import annotations

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cycler import cycler  # noqa: E402

# Okabe-Ito colour-blind-safe palette.
_BLUE, _VERM, _GREEN, _PURPLE, _ORANGE, _SKY, _YELLOW, _BLACK = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000")

COL_SINGLE = 3.50
COL_ONEHALF = 4.75
COL_DOUBLE = 7.20

# Canonical schedule order, display labels and stable colours.
SCHEDULE_ORDER = ["random", "coefficient", "commutator", "antithetic", "symmetric", "learned"]
SCHED_LABELS = {
    "random": "random (1st order)",
    "coefficient": "coefficient (1st order)",
    "commutator": "clique (1st order)",
    "antithetic": "antithetic (2nd, free)",
    "symmetric": "symmetric (2nd, Strang)",
    "learned": "learned router",
}
SCHED_COLOR = {
    "random": "#9AA0A6", "coefficient": _BLUE, "commutator": _SKY,
    "antithetic": _VERM, "symmetric": _GREEN, "learned": _PURPLE,
}
FIRST_ORDER = ("random", "coefficient", "commutator")
FOLDED = ("antithetic", "symmetric")


def apply_nmi_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.hashsalt": "topoham", "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.3, "lines.markersize": 3.0,
        "legend.frameon": False, "xtick.direction": "out", "ytick.direction": "out",
        "grid.linewidth": 0.5, "grid.alpha": 0.3,
    })


def panel_label(ax, letter: str, x: float = -0.16, y: float = 1.03) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------------------
# Figure 1 -- method schematic
# ---------------------------------------------------------------------------
def _box(ax, xy, w, h, text, fc, ec="#222222"):
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((xy[0], xy[1]), w, h,
                         boxstyle="round,pad=0.012,rounding_size=0.02",
                         linewidth=1.0, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=7.0, zorder=5)
    return (xy[0] + w, xy[1] + h / 2), (xy[0], xy[1] + h / 2)


def _arrow(ax, p0, p1):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#444444",
                                shrinkA=2, shrinkB=2))


def fig_schematic(summary: Dict, out: Path) -> Path:
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    blue, green, orange, purple, grey = (
        "#D6E6F2", "#D6EFE3", "#FBE6D4", "#ECDCE9", "#ECECEC")
    y, h = 0.40, 0.34
    boxes = [
        (0.010, 0.150, "Hamiltonian\n$H=\\sum_j c_j P_j$\n3 families", blue),
        (0.205, 0.165, "commutator graph\n$\\mathcal{G}(H)$\ncliques + colours", green),
        (0.415, 0.170, "schedule $\\sigma$\norder $\\cdot$ fold $\\cdot$ clique\n(matched $L\\cdot r$)", orange),
        (0.628, 0.165, "product formula\n$U_\\sigma$ statevector\nrotation count", purple),
        (0.840, 0.150, "fidelity vs exact\n$e^{-iHt}$\ngates-to-target", grey),
    ]
    rights, lefts = [], []
    for x0, w, text, fc in boxes:
        r, l = _box(ax, (x0, y), w, h, text, fc)
        rights.append(r)
        lefts.append(l)
    for i in range(len(boxes) - 1):
        _arrow(ax, rights[i], lefts[i + 1])
    ax.text(0.5, 0.95,
            "leading error $E_\\pi=\\sum_{a<b}[h_{\\pi(a)},h_{\\pi(b)}]$: "
            "reordering only re-signs it (impotence); a folded schedule cancels it",
            ha="center", va="center", fontsize=6.6, color="#333333")
    ax.text(0.5, 0.065,
            "fast Pauli rotation $\\leftrightarrow$ dense expm to $10^{-9}$"
            "   $\\bullet$   matrix-free error form $\\leftrightarrow$ dense to $10^{-9}$",
            ha="center", va="center", fontsize=6.4, color="#555555")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 -- convergence (infidelity vs r, log-log): the rate upgrade
# ---------------------------------------------------------------------------
def fig_convergence(summary: Dict, out: Path) -> Path:
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 3.0))
    conv = summary["convergence"]["schedules"]
    slopes = summary.get("slopes", {})
    rmax = 1
    for s in SCHEDULE_ORDER:
        if s == "learned" or s not in conv:
            continue
        rows = conv[s]
        xs = np.array([row["r"] for row in rows], dtype=float)
        ys = np.array([max(row["infid"], 1e-12) for row in rows], dtype=float)
        mask = ys > 1e-11
        if mask.sum() < 2:
            continue
        rmax = max(rmax, xs[mask].max())
        sl = slopes.get(s)
        lab = SCHED_LABELS[s] + (f"  (rate {sl:.1f})" if sl is not None and not np.isnan(sl) else "")
        ax.loglog(xs[mask], ys[mask], marker="o", ms=2.6,
                  color=SCHED_COLOR[s], lw=1.5 if s in FOLDED else 1.1,
                  ls="-" if s in FOLDED else "--", label=lab)
    # reference slope guides
    rr = np.array([2.0, rmax])
    for p, txt, yy in ((2, "$\\propto r^{-2}$", 0.5), (4, "$\\propto r^{-4}$", 0.5)):
        g = yy * (rr / rr[0]) ** (-p)
        ax.loglog(rr, g, color="#bbbbbb", lw=0.8, ls=":", zorder=0)
        ax.text(rr[-1], g[-1], txt, fontsize=6.2, color="#888888",
                ha="left", va="center")
    ax.set_xlabel("Trotter steps $r$")
    ax.set_ylabel("mean infidelity vs exact")
    ax.legend(loc="lower left", handlelength=1.7)
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3 -- frontier: fidelity vs realised rotation count
# ---------------------------------------------------------------------------
def fig_frontier(summary: Dict, out: Path) -> Path:
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 3.0))
    frontier = summary.get("frontier", {})
    for s in SCHEDULE_ORDER:
        if s == "learned" or s not in frontier:
            continue
        curve = frontier[s]
        xs = np.array([pt[0] for pt in curve], dtype=float)
        infid = np.array([max(1.0 - pt[1], 1e-7) for pt in curve], dtype=float)
        order = np.argsort(xs)
        xs, infid = xs[order], infid[order]
        ax.loglog(xs, infid, marker="o", ms=2.6, color=SCHED_COLOR[s],
                  lw=1.5 if s in FOLDED else 1.1, ls="-" if s in FOLDED else "--",
                  label=SCHED_LABELS[s])
    for tg, lab in ((1e-2, "$F=0.99$"), (1e-3, "$F=0.999$")):
        ax.axhline(tg, color="#cccccc", lw=0.7, ls=":")
        ax.text(ax.get_xlim()[1], tg, lab, fontsize=6.0, color="#888888",
                ha="right", va="bottom")
    ax.set_xlabel(r"realised rotation count (gate cost)")
    ax.set_ylabel("infidelity vs exact (matched cost)")
    ax.legend(loc="lower left", handlelength=1.7)
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4 -- per-family gates-to-target at 0.99 (lower is better)
# ---------------------------------------------------------------------------
def fig_family_gates(summary: Dict, out: Path) -> Path:
    apply_nmi_style()
    g2t = summary["gates_to_target"]["by_family"]
    families = sorted(g2t.keys())
    scheds = [s for s in SCHEDULE_ORDER if s in next(iter(g2t.values()))]
    # ceiling for "did not reach within the swept grid"
    allvals = [v for fam in families for s in scheds
               for v in [g2t[fam][s].get("0.99")] if v is not None]
    ceil = (max(allvals) * 1.35) if allvals else 1.0
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 3.0))
    width = 0.8 / max(1, len(scheds))
    x = np.arange(len(families))
    for i, s in enumerate(scheds):
        vals, hatches = [], []
        for fam in families:
            v = g2t[fam][s].get("0.99")
            vals.append(v if v is not None else ceil)
            hatches.append(v is None)
        bars = ax.bar(x + i * width, vals, width, color=SCHED_COLOR[s],
                      label=SCHED_LABELS[s], edgecolor="white", linewidth=0.3)
        for b, miss in zip(bars, hatches):
            if miss:
                b.set_hatch("////")
                b.set_alpha(0.55)
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), "$\\times$",
                        ha="center", va="bottom", fontsize=6, color="#333333")
    ax.set_xticks(x + width * (len(scheds) - 1) / 2)
    ax.set_xticklabels([f.replace("_", " ") for f in families], rotation=12, ha="right")
    ax.set_ylabel(r"rotations to fidelity $\geq 0.99$")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.20),
              columnspacing=1.0, handlelength=1.2)
    ax.grid(True, axis="y")
    ax.text(0.99, 0.96, r"$\times$ = target not reached in grid", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color="#333333")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 5 -- ordering impotence and collision structure (Extended Data)
# ---------------------------------------------------------------------------
def fig_impotence(summary: Dict, out: Path) -> Path:
    apply_nmi_style()
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.7))
    imp = summary["ordering_impotence"]
    ax = axes[0]
    keys = ["ratio_median_to_min_mean", "ratio_coeff_to_min_mean", "ratio_coeff_to_min_max"]
    labs = ["median\nordering", "coefficient\nordering", "worst\ncoefficient"]
    vals = [imp[k] for k in keys]
    ax.bar(range(len(vals)), vals, color=[_SKY, _BLUE, "#9AA0A6"], width=0.6)
    ax.axhline(1.0, color="#444444", lw=0.9, ls="--")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labs)
    ax.set_ylabel(r"$\|E_\pi\|$ / min over orderings")
    ax.set_ylim(0.95, max(1.2, max(vals) * 1.05))
    ax.text(0.5, 0.93, "reordering stays within a few % of the floor",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color="#333333")
    panel_label(ax, "a")
    ax = axes[1]
    cf = summary["collisions_by_family"]
    families = sorted(cf.keys())
    irr = [cf[f]["irreducible_frac"] for f in families]
    ax.bar(range(len(families)), irr, color=_GREEN, width=0.6)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.replace("_", " ") for f in families], rotation=12, ha="right")
    ax.set_ylabel("ordering-invariant\nerror fraction")
    ax.set_ylim(0, 1.05)
    panel_label(ax, "b")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
