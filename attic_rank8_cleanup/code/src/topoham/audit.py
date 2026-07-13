"""Claim auditing: no fabricated numbers, no forbidden superiority claims.

Implements the readiness gate from COPILOT_INSTRUCTIONS:
  * every claimed number must trace to a value in the generated summary;
  * the mandated reproducibility sentence must be present;
  * forbidden absolute-superiority phrasing is rejected unless evidenced.
"""
from __future__ import annotations

import re
from typing import Dict, List

REQUIRED_PHRASE = (
    "All experiments are reproducible on commodity hardware; "
    "runtime and memory are reported for each benchmark."
)

FORBIDDEN_PATTERNS = [
    r"state[- ]of[- ]the[- ]art",
    r"\bworld[- ]?class\b",
    r"\bbest[- ]ever\b",
    r"\bguaranteed optimal\b",
    r"\bbeats all\b",
    r"\boutperforms everyone\b",
]


def audit_reproducibility_phrase(text: str) -> List[str]:
    return [] if REQUIRED_PHRASE in text else [
        "missing required reproducibility phrase"
    ]


def audit_forbidden_claims(text: str) -> List[str]:
    problems = []
    for pat in FORBIDDEN_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            problems.append(f"forbidden claim {m.group(0)!r}")
    return problems


def audit_numbers(claimed: Dict[str, float], summary: Dict[str, float],
                  tol: float = 1e-6) -> List[str]:
    """Each claimed macro must equal a generated summary value."""
    problems = []
    for key, val in claimed.items():
        if key not in summary:
            problems.append(f"claimed number {key!r} has no generated artifact")
        elif abs(summary[key] - val) > tol:
            problems.append(
                f"claimed {key}={val} != generated {summary[key]}"
            )
    return problems
