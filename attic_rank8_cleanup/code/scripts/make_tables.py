#!/usr/bin/env python
"""Generate LaTeX macros and table fragments from results/summary.json.

Writes, into ``results/``:
  * ``macros.tex``      -- \\newcommand definitions for every number quoted in
                           the manuscript prose (so the text is traceable);
  * ``tab_matched.tex`` -- matched-budget snapshot at the reference step count;
  * ``tab_gates.tex``   -- gates-to-target-fidelity (the headline efficiency);
  * ``tab_family.tex``  -- per-family gates-to-target;
  * ``tab_impotence.tex``-- ordering-impotence + collision structure (Theorem 1).
The manuscript \\input's these fragments, so no number is entered by hand.
"""
import json
from pathlib import Path

import _bootstrap  # noqa: F401

SCHED = ["random", "coefficient", "commutator", "antithetic", "symmetric"]
DISP = {"random": "random (1st)", "coefficient": "coefficient (1st)",
        "commutator": "clique (1st)", "antithetic": "antithetic (2nd, free)",
        "symmetric": "symmetric (2nd)", "learned": "learned router"}


def _g(v):
    return "---" if v is None else str(int(v))


def main() -> None:
    s = json.loads(Path("results/summary.json").read_text())
    out = Path("results")
    out.mkdir(exist_ok=True)
    h = s["headline"]
    st = s["schedules_table"]
    g2t = s["gates_to_target"]
    imp = s["ordering_impotence"]
    pig = s["per_instance_gates_to_target"]
    r0 = s["config"]["steps"]
    gates_r0 = int(round(st["coefficient"]["gates_mean"]))

    # ---- macros -----------------------------------------------------------
    def cmd(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}"
    m = [
        "% Auto-generated from results/summary.json by scripts/make_tables.py.",
        cmd("Ninstances", h["n_instances"]),
        cmd("EvolTime", f"{s['config']['time']:.1f}"),
        cmd("RefSteps", r0),
        cmd("RefGates", gates_r0),
        cmd("GtFirst", _g(h["gates_to_target_first_order"])),
        cmd("GtAnti", _g(h["gates_to_target_antithetic"])),
        cmd("GtSym", _g(h["gates_to_target_symmetric"])),
        cmd("GateSpeedup", f"{h['gate_speedup_vs_first_order']:.2f}"),
        cmd("SlopeFirst", f"{h['slope_first_order']:.2f}"),
        cmd("SlopeAnti", f"{h['slope_antithetic']:.2f}"),
        cmd("SlopeSym", f"{h['slope_symmetric']:.2f}"),
        cmd("ImpMedian", f"{imp['ratio_median_to_min_mean']:.3f}"),
        cmd("ImpCoeff", f"{imp['ratio_coeff_to_min_mean']:.3f}"),
        cmd("ImpMax", f"{imp['ratio_coeff_to_min_max']:.2f}"),
        cmd("ImpHS", f"{imp['hs_ratio_max_to_min_mean']:.3f}"),
        cmd("FidFirstRef", f"{st['coefficient']['fidelity']['mean']:.4f}"),
        cmd("FidAntiRef", f"{st['antithetic']['fidelity']['mean']:.4f}"),
        cmd("InfFirstRef", f"{st['coefficient']['infidelity_mean']:.4f}"),
        cmd("InfAntiRef", f"{st['antithetic']['infidelity_mean']:.4f}"),
        cmd("InfRatioRef", f"{st['coefficient']['infidelity_mean']/max(st['antithetic']['infidelity_mean'],1e-9):.1f}"),
        cmd("GtFirstHard", _g(g2t["overall"]["coefficient"]["0.999"])),
        cmd("GtAntiHard", _g(g2t["overall"]["antithetic"]["0.999"])),
        cmd("GtSymHard", _g(g2t["overall"]["symmetric"]["0.999"])),
        cmd("LearnedAcc", f"{s['learned']['leave_one_out_accuracy']*100:.0f}"),
        cmd("LearnedMeanGates", f"{pig['mean_gates_common'].get('learned')}"),
        cmd("OracleMeanGates", f"{pig['oracle_mean_gates_common']}"),
        cmd("SymMeanGates", f"{pig['mean_gates_common'].get('symmetric')}"),
        cmd("AntiMeanGates", f"{pig['mean_gates_common'].get('antithetic')}"),
        cmd("CommonN", pig["common_subset_n"]),
        cmd("IrrTFIM", f"{s['collisions_by_family']['tfim']['irreducible_frac']:.2f}"),
        cmd("IrrHeis", f"{s['collisions_by_family']['heisenberg']['irreducible_frac']:.2f}"),
        cmd("IrrMol", f"{s['collisions_by_family']['molecular_like']['irreducible_frac']:.2f}"),
        cmd("Runtime", f"{s['provenance']['runtime_sec']:.0f}"),
        cmd("PeakMem", f"{s['provenance']['peak_memory_mb']:.0f}"),
    ]
    (out / "macros.tex").write_text("\n".join(m) + "\n")

    # ---- Table 1: matched-budget snapshot at r0 ---------------------------
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"schedule & order & fidelity & $\pm$95\% & rotations & rate $p$ \\",
             r"\midrule"]
    for sc in SCHED:
        f = st[sc]["fidelity"]
        order = "1st" if sc in ("random", "coefficient", "commutator") else "2nd"
        lines.append(f"{DISP[sc]} & {order} & {f['mean']:.4f} & {f['ci95']:.4f} & "
                     f"{int(round(st[sc]['gates_mean']))} & {st[sc]['slope']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_matched.tex").write_text("\n".join(lines) + "\n")

    # ---- Table 2: gates-to-target (overall) -------------------------------
    tg = g2t["targets"]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             "schedule & " + " & ".join(f"$F\\!\\geq\\!{t}$" for t in tg) + r" \\",
             r"\midrule"]
    for sc in SCHED:
        cells = " & ".join(_g(g2t["overall"][sc][f"{t}"]) for t in tg)
        lines.append(f"{DISP[sc]} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_gates.tex").write_text("\n".join(lines) + "\n")

    # ---- Table 3: per-family gates-to-target at 0.99 and 0.999 ------------
    fams = sorted(g2t["by_family"].keys())
    lines = [r"\begin{tabular}{ll" + "cc" + r"}", r"\toprule",
             r"family & schedule & $F\!\geq\!0.99$ & $F\!\geq\!0.999$ \\",
             r"\midrule"]
    for fam in fams:
        for k, sc in enumerate(SCHED):
            fname = fam.replace("_", "\\_") if k == 0 else ""
            row = g2t["by_family"][fam][sc]
            lines.append(f"{fname} & {DISP[sc]} & {_g(row['0.99'])} & {_g(row['0.999'])} \\\\")
        if fam != fams[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_family.tex").write_text("\n".join(lines) + "\n")

    # ---- Table 4: ordering impotence + collision structure ----------------
    cf = s["collisions_by_family"]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"family & anticomm.\ pairs & colliding & ordering-invariant frac. \\",
             r"\midrule"]
    for fam in sorted(cf.keys()):
        c = cf[fam]
        lines.append(f"{fam.replace('_', chr(92)+'_')} & {c['n_anti_pairs']:.1f} & "
                     f"{c['colliding_pairs']:.1f} & {c['irreducible_frac']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_impotence.tex").write_text("\n".join(lines) + "\n")

    print("wrote results/macros.tex, tab_matched.tex, tab_gates.tex, "
          "tab_family.tex, tab_impotence.tex")


if __name__ == "__main__":
    main()
