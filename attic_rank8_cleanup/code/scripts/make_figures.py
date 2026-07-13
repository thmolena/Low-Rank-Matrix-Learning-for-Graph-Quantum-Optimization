#!/usr/bin/env python
"""Generate all figures from results/summary.json."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from topoham import plotting


def main() -> None:
    summary = json.loads(Path("results/summary.json").read_text())
    out = Path("figures")
    out.mkdir(exist_ok=True)
    plotting.fig_schematic(summary, out / "fig_schematic.pdf")
    plotting.fig_convergence(summary, out / "fig_convergence.pdf")
    plotting.fig_frontier(summary, out / "fig_frontier.pdf")
    plotting.fig_family_gates(summary, out / "fig_family.pdf")
    plotting.fig_impotence(summary, out / "fig_impotence.pdf")
    print(f"wrote {out}/fig_schematic.pdf, {out}/fig_convergence.pdf, "
          f"{out}/fig_frontier.pdf, {out}/fig_family.pdf, {out}/fig_impotence.pdf")


if __name__ == "__main__":
    main()
