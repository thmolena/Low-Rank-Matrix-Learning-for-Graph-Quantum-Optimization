#!/usr/bin/env python
"""Run the Trotter term-ordering experiment and write results/summary.json."""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from topoham.config import Config
from topoham.runner import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    summary = run(cfg, Path(args.out))
    h = summary["headline"]
    print(f"[{cfg.name}] instances={h['n_instances']}  target={h['target']}  "
          f"gates-to-target: first-order={h['gates_to_target_first_order']} "
          f"antithetic={h['gates_to_target_antithetic']} "
          f"symmetric={h['gates_to_target_symmetric']}  "
          f"(speedup {h['gate_speedup_vs_first_order']}x)  "
          f"learned/oracle mean gates (common)={h['learned_mean_gates_common']}/"
          f"{h['oracle_mean_gates_common']} vs symmetric "
          f"{h['symmetric_mean_gates_common']}  "
          f"slopes: 1st={h['slope_first_order']} anti={h['slope_antithetic']} "
          f"sym={h['slope_symmetric']}  "
          f"impotence-ratio={h['ordering_impotence_ratio']}  "
          f"runtime={summary['provenance']['runtime_sec']}s")


if __name__ == "__main__":
    main()
