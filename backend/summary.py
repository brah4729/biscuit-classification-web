"""
summary.py — Turns rules.evaluate_sample() output into a plain-language
"Executive Summary", matching the proposal's promise of an auto-generated
report that doesn't require the reader to be a data scientist.

Deliberately template-based, not an LLM call:
- No API cost, no latency, no chance of the model saying something wrong
  in front of judges.
- Fully deterministic — same input always gives the same summary, which
  matters for a system that's meant to support certification decisions.
"""


def generate_summary(sample: dict, result: dict) -> str:
    """
    sample: the raw input dict, e.g. {"moisture": 6.2, "protein": 6.0}
    result: the dict returned by rules.evaluate_sample(sample)

    Returns a short paragraph of plain-language text.
    """
    if result["passed"]:
        checked = ", ".join(result["checked_parameters"]) or "no parameters"
        return (
            f"PASS — This sample meets all evaluated SNI 2973:2011 "
            f"requirements ({checked}). No corrective action needed."
        )

    lines = [
        f"FAIL — This sample did not meet {len(result['failures'])} "
        f"of {len(result['checked_parameters'])} evaluated quality requirement(s)."
    ]

    # List each specific failure with its reason — this is the part that
    # makes it "actionable" rather than just a pass/fail stamp.
    for f in result["failures"]:
        lines.append(f"  • {f['reason']}")

    # Group-level insight: which category of problem is this, broadly?
    # This is the "clustering-flavored" part — telling the auditor not just
    # WHAT failed, but WHAT KIND of problem it represents.
    categories = result["failure_categories"]
    if len(categories) == 1:
        lines.append(
            f"This failure pattern is classified as a "
            f"'{categories[0]}' issue, suggesting the root cause is "
            f"isolated to one part of the production/quality process."
        )
    else:
        joined = ", ".join(categories)
        lines.append(
            f"This sample shows failures across multiple categories "
            f"({joined}), suggesting compounding issues rather than a "
            f"single root cause — recommend a full batch review rather "
            f"than a single-parameter fix."
        )

    return "\n".join(lines)
