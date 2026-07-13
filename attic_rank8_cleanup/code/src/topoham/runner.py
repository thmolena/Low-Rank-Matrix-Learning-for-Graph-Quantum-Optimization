"""End-to-end experiment driver for the scheduling study.

Produces ``results/summary.json`` -- the single source of truth for every table,
figure and macro. The protocol, for each Hamiltonian instance:

  1. build the exact reference  e^{-iHt}|psi0>  (dense expm or Krylov);
  2. for every schedule and every step count r in the grid, run the product
     formula and score fidelity at the *realised rotation count* (the gate cost);
  3. from the (gates, fidelity) curves compute the gates-to-target-fidelity (the
     headline matched-cost efficiency metric) and the fidelity-vs-gate frontier;
  4. quantify ordering impotence -- how little any reordering moves the leading
     commutator-error operator norm (Theorem 1) -- and fit the convergence rate
     of each schedule (first-order ~1/r^2 vs folded ~1/r^4, Theorem 2).

Schedules generalise term orderings: ``random``/``coefficient``/``commutator`` are
first-order; ``antithetic`` is the zero-overhead step-alternating fold (identical
L*r rotations) and ``symmetric`` is the Strang half-step fold. The ``learned``
policy is a router over the fixed schedules, trained leave-one-instance-out.

Two integrity flags are recorded before any number is emitted: the fast
Pauli-rotation kernel matches dense ``expm`` (``pauli_algebra_verified``) and the
matrix-free commutator error form matches the dense pair-commutator sum
(``error_form_verified``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.linalg import expm

from . import error_form, metrics, pauli, policies, schedules
from .config import Config
from .env import TrotterEnv
from .hamiltonians import Hamiltonian, build
from .policies import LearnedSchedulePolicy, feature_vector
from .seed import RunProvenance, set_seed

FIXED = list(schedules.FIXED_SCHEDULES)
ALL_SCHEDULES = FIXED + ["learned"]


@dataclass
class Instance:
    family: str
    n: int
    ham: Hamiltonian


def _verify_pauli_algebra(rng: np.random.Generator) -> bool:
    """Cross-check the fast Pauli-rotation kernel against dense ``expm``."""
    ok = True
    for n in (2, 3):
        for _ in range(8):
            p = "".join(rng.choice(list("IXYZ"), size=n))
            theta = float(rng.uniform(-np.pi, np.pi))
            psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
            psi /= np.linalg.norm(psi)
            fast = pauli.apply_pauli_rotation(psi, p, theta)
            exact = expm(-1j * theta * pauli.to_matrix(p)) @ psi
            if not np.allclose(fast, exact, atol=1e-9):
                ok = False
    return ok


def _gates_to_target(curve: Dict[int, tuple], target: float) -> Optional[int]:
    """First (smallest-r) rotation count at which mean fidelity reaches ``target``.

    ``curve`` maps r -> (mean_gates, mean_fidelity). Scanning by ascending r keeps
    the matched-cost reading monotone in the controllable budget knob.
    """
    for r in sorted(curve):
        g, f = curve[r]
        if f >= target:
            return int(round(g))
    return None


def _fit_slope(rs: List[int], infids: List[float], even_only: bool = True) -> float:
    """Log-log convergence rate -d(log infidelity)/d(log r) over the asymptotic
    (upper) part of the grid: first-order ~2, second-order folds ~4. Restricting
    to even r keeps the antithetic fold (whose step-pairs are palindromic only for
    even r) in its clean second-order regime."""
    pts = [(r, i) for r, i in zip(rs, infids)
           if i > 1e-11 and i < 0.5 and r > 1 and (r % 2 == 0 or not even_only)]
    pts = pts[len(pts) // 3:] if len(pts) >= 4 else pts
    if len(pts) < 2:
        return float("nan")
    x = np.log(np.array([r for r, _ in pts], dtype=float))
    y = np.log(np.array([i for _, i in pts], dtype=float))
    return float(-np.polyfit(x, y, 1)[0])


def _mean_instance_slope(per_inst, idxs, grid, s):
    """Mean over instances of the per-instance convergence rate (robust to mixing
    exactly-solved instances, which are skipped)."""
    vals = []
    for i in idxs:
        infids = [1.0 - per_inst[i][s][r][0] for r in grid]
        sl = _fit_slope(grid, infids)
        if not math.isnan(sl):
            vals.append(sl)
    return round(float(np.mean(vals)), 3) if vals else float("nan")


def _best_schedule_label(rec: Dict[str, Dict[int, tuple]], target: float) -> str:
    """The fixed schedule reaching ``target`` at fewest rotations on this instance
    (or, if none reach it, the one with the highest attained fidelity).

    ``rec[s]`` maps r -> (fidelity, gates) for a single instance.
    """
    best, best_key = FIXED[0], (2, float("inf"), float("inf"))
    for s in FIXED:
        curve = rec[s]
        reached = None
        for r in sorted(curve):
            f, gates = curve[r]
            if f >= target:
                reached = gates
                break
        if reached is not None:
            key = (0, int(reached), 0.0)
        else:
            max_fid = max(f for f, _ in curve.values())
            min_gates = min(g for _, g in curve.values())
            key = (1, -max_fid, int(min_gates))
        if key < best_key:
            best, best_key = s, key
    return best


def run(cfg: Config, out_dir: Path) -> Dict:
    prov = RunProvenance(seed=cfg.seed)
    rng = set_seed(cfg.seed)

    pauli_ok = _verify_pauli_algebra(rng)
    form_ok = error_form.verify_error_form(np.random.default_rng(cfg.seed + 1))

    # 1. Build the instance pool.
    instances: List[Instance] = []
    for fam in cfg.families:
        for n in cfg.sizes:
            for _ in range(cfg.n_per_family):
                ham = build(fam, n, rng)
                instances.append(Instance(fam, n, ham))

    grid = list(cfg.steps_grid)
    r0 = cfg.steps

    # 2. Per-instance (schedule, r) -> (fidelity, gates); plus impotence + features.
    per_inst: List[Dict[str, Dict[int, tuple]]] = []
    impotence_rows: List[Dict[str, float]] = []
    feats = [feature_vector(inst.ham) for inst in instances]
    for i, inst in enumerate(instances):
        env = TrotterEnv.from_hamiltonian(inst.ham, cfg.time, 1,
                                          reference_backend=cfg.reference_backend)
        rec: Dict[str, Dict[int, tuple]] = {s: {} for s in FIXED}
        for s in FIXED:
            for r in grid:
                res = env.evaluate_schedule(s, r, np.random.default_rng(cfg.seed + i))
                rec[s][r] = (res["fidelity"], res["gates"])
        per_inst.append(rec)
        impotence_rows.append(
            error_form.ordering_impotence(inst.ham, np.random.default_rng(cfg.seed + 7 + i),
                                          n_samples=cfg.impotence_samples))

    # 3. Leave-one-instance-out learned router over schedules.
    labels = [_best_schedule_label(rec, cfg.target_ref) for rec in per_inst]
    learned_choice: List[str] = []
    for i in range(len(instances)):
        train_idx = [k for k in range(len(instances)) if k != i]
        router = LearnedSchedulePolicy().fit(
            [instances[k].ham for k in train_idx], [labels[k] for k in train_idx])
        learned_choice.append(router.predict_schedule(instances[i].ham))

    # Attach the learned schedule's own curve to each instance record.
    for i, rec in enumerate(per_inst):
        rec["learned"] = dict(rec[learned_choice[i]])

    # 4. Aggregate (schedule, r) -> mean gates, mean/ci fidelity, mean infidelity.
    def agg(idxs: List[int]):
        out: Dict[str, Dict[int, dict]] = {s: {} for s in ALL_SCHEDULES}
        for s in ALL_SCHEDULES:
            for r in grid:
                fids = np.array([per_inst[i][s][r][0] for i in idxs], dtype=float)
                gates = np.array([per_inst[i][s][r][1] for i in idxs], dtype=float)
                n = max(1, fids.size)
                ci = 1.96 * float(fids.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
                out[s][r] = {
                    "gates": float(gates.mean()),
                    "fid": float(fids.mean()), "fid_ci": ci,
                    "infid": float((1.0 - fids).mean()),
                }
        return out

    all_idx = list(range(len(instances)))
    agg_all = agg(all_idx)
    by_fam_idx: Dict[str, List[int]] = {}
    for i, inst in enumerate(instances):
        by_fam_idx.setdefault(inst.family, []).append(i)
    agg_fam = {fam: agg(idxs) for fam, idxs in by_fam_idx.items()}

    def curve_of(a, s):  # r -> (gates, fid) for gates-to-target
        return {r: (a[s][r]["gates"], a[s][r]["fid"]) for r in grid}

    # Headline: gates-to-target-fidelity over the aggregate curve. Computed for
    # the FIXED schedules only -- they have a consistent rotation count at each r,
    # so the matched-cost reading is well defined. The learned router (whose pick
    # varies by instance) is reported per-instance below instead.
    gates_to_target = {
        "targets": list(cfg.targets),
        "overall": {s: {f"{tg}": _gates_to_target(curve_of(agg_all, s), tg)
                        for tg in cfg.targets} for s in FIXED},
        "by_family": {fam: {s: {f"{tg}": _gates_to_target(curve_of(a, s), tg)
                                for tg in cfg.targets} for s in FIXED}
                      for fam, a in agg_fam.items()},
    }

    # Per-instance gates-to-target at the reference target: the fair way to score
    # the instance-adaptive learned router against the per-instance oracle and the
    # fixed schedules, on the subset every fixed schedule reaches.
    def inst_g2t(rec_s):
        for r in grid:
            f, g = rec_s[r]
            if f >= cfg.target_ref:
                return int(g)
        return None
    ig = {s: [inst_g2t(per_inst[i][s]) for i in all_idx]
          for s in ALL_SCHEDULES}
    oracle = []
    for k in range(len(instances)):
        cand = [ig[s][k] for s in FIXED if ig[s][k] is not None]
        oracle.append(min(cand) if cand else None)
    common = [k for k in range(len(instances)) if all(ig[s][k] is not None for s in FIXED)]
    per_instance_g2t = {
        "target": cfg.target_ref,
        "n_instances": len(instances),
        "common_subset_n": len(common),
        "mean_gates_common": {
            s: round(float(np.mean([ig[s][k] for k in common])), 1) if common else None
            for s in ALL_SCHEDULES},
        "oracle_mean_gates_common": round(float(np.mean([oracle[k] for k in common])), 1)
        if common else None,
        "reach_rate": {
            s: round(float(np.mean([ig[s][k] is not None for k in all_idx])), 3)
            for s in ALL_SCHEDULES},
    }

    # Convergence (infidelity vs r) + fitted convergence rates (Theorem 2).
    # The reported rate is the mean per-instance log-log slope (robust to mixing
    # exactly-solved instances); the aggregate curve is kept for the figure.
    convergence = {"r_grid": grid, "schedules": {}}
    slopes = {}
    slopes_by_family = {}
    for s in ALL_SCHEDULES:
        rows = [{"r": r, "gates": round(agg_all[s][r]["gates"], 2),
                 "fid": round(agg_all[s][r]["fid"], 6),
                 "fid_ci": round(agg_all[s][r]["fid_ci"], 6),
                 "infid": round(agg_all[s][r]["infid"], 8)} for r in grid]
        convergence["schedules"][s] = rows
        slopes[s] = _mean_instance_slope(per_inst, all_idx, grid, s)
        slopes_by_family[s] = {fam: _mean_instance_slope(per_inst, idxs, grid, s)
                               for fam, idxs in by_fam_idx.items()}

    # Frontier (fidelity vs realised total rotation count).
    frontier = {s: [[int(round(agg_all[s][r]["gates"])),
                     round(agg_all[s][r]["fid"], 6),
                     round(agg_all[s][r]["fid_ci"], 6)] for r in grid]
                for s in ALL_SCHEDULES}

    # Matched-budget snapshot at the reference step count r0.
    sched_table = {}
    for s in ALL_SCHEDULES:
        fids = np.array([per_inst[i][s][r0][0] for i in all_idx], dtype=float)
        gates = np.array([per_inst[i][s][r0][1] for i in all_idx], dtype=float)
        sched_table[s] = {
            "fidelity": metrics.summarize(fids.tolist()),
            "infidelity_mean": round(float((1.0 - fids).mean()), 6),
            "gates_mean": round(float(gates.mean()), 2),
            "slope": slopes[s],
        }

    by_family = {fam: {s: metrics.summarize([per_inst[i][s][r0][0] for i in idxs])
                       for s in ALL_SCHEDULES}
                 for fam, idxs in by_fam_idx.items()}

    # Ordering impotence (Theorem 1): how little reordering moves ||E_pi||.
    def col(key):
        return np.array([row[key] for row in impotence_rows], dtype=float)
    imp = {
        "ratio_coeff_to_min_mean": round(float(col("ratio_coeff_to_min").mean()), 4),
        "ratio_median_to_min_mean": round(float(col("ratio_median_to_min").mean()), 4),
        "ratio_coeff_to_min_max": round(float(col("ratio_coeff_to_min").max()), 4),
        "hs_ratio_max_to_min_mean": round(float(col("hs_ratio_max_to_min").mean()), 4),
    }
    # Collision structure per family (the resource that *is* ordering-reducible).
    collisions_by_family = {}
    for fam, idxs in by_fam_idx.items():
        cs = [error_form.collision_stats(instances[i].ham) for i in idxs]
        collisions_by_family[fam] = {
            "n_anti_pairs": round(float(np.mean([c["n_anti_pairs"] for c in cs])), 2),
            "colliding_pairs": round(float(np.mean([c["colliding_pairs"] for c in cs])), 2),
            "irreducible_frac": round(float(np.mean([c["irreducible_frac"] for c in cs])), 4),
        }

    # Learned router agreement with the per-instance optimum.
    learned_acc = round(float(np.mean([learned_choice[i] == labels[i]
                                       for i in all_idx])), 4)

    summary: Dict = {
        "config": cfg.__dict__,
        "provenance": prov.finalize().to_dict(),
        "pauli_algebra_verified": bool(pauli_ok),
        "error_form_verified": bool(form_ok),
        "schedules_table": sched_table,
        "gates_to_target": gates_to_target,
        "per_instance_gates_to_target": per_instance_g2t,
        "convergence": convergence,
        "slopes": slopes,
        "slopes_by_family": slopes_by_family,
        "frontier": frontier,
        "by_family": by_family,
        "ordering_impotence": imp,
        "collisions_by_family": collisions_by_family,
        "learned": {
            "label_distribution": {s: int(labels.count(s)) for s in FIXED},
            "choice_distribution": {s: int(learned_choice.count(s)) for s in FIXED},
            "leave_one_out_accuracy": learned_acc,
        },
        "headline": {},
    }

    # Headline macros.
    tgt = cfg.target_ref
    g_first = gates_to_target["overall"]["coefficient"][f"{tgt}"]
    g_anti = gates_to_target["overall"]["antithetic"][f"{tgt}"]
    g_sym = gates_to_target["overall"]["symmetric"][f"{tgt}"]
    best_folded = min([g for g in (g_anti, g_sym) if g is not None], default=None)
    speedup = (round(g_first / best_folded, 2)
               if (g_first and best_folded) else None)
    summary["headline"] = {
        "n_instances": len(instances),
        "time": cfg.time,
        "target": tgt,
        "gates_to_target_first_order": g_first,
        "gates_to_target_antithetic": g_anti,
        "gates_to_target_symmetric": g_sym,
        "gate_speedup_vs_first_order": speedup,
        "learned_mean_gates_common": per_instance_g2t["mean_gates_common"].get("learned"),
        "oracle_mean_gates_common": per_instance_g2t["oracle_mean_gates_common"],
        "symmetric_mean_gates_common": per_instance_g2t["mean_gates_common"].get("symmetric"),
        "learned_reach_rate": per_instance_g2t["reach_rate"].get("learned"),
        "slope_first_order": slopes["coefficient"],
        "slope_antithetic": slopes["antithetic"],
        "slope_symmetric": slopes["symmetric"],
        "ordering_impotence_ratio": imp["ratio_median_to_min_mean"],
        "error_form_verified": bool(form_ok),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
