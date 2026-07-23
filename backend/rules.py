"""
rules.py — Rule-based biscuit lab quality engine.

Why rule-based instead of a trained ML model:
No real or committee-provided lab dataset exists for this case as of the
submission deadline. A model "trained" on fabricated data would report a
meaningless accuracy number. A rule engine encoding real SNI 2973:2011
thresholds is fully explainable — an auditor can point to exactly which
number, on which standard, caused a failure. See AGENTS.md Section 3.

This file has NO Flask/web code in it on purpose. It's plain Python, so
you can test it directly:

    python3
    >>> from rules import evaluate_sample
    >>> evaluate_sample({"moisture": 6.2, "protein": 6.0, "ash": 0.8})
"""

# ─── Threshold Config ──────────────────────────────────────────────────────
# Each entry describes ONE lab parameter and how to judge it.
#
#   limit       -> the numeric cutoff
#   comparison  -> "max" means "fail if value is ABOVE limit"
#                  "min" means "fail if value is BELOW limit"
#   unit        -> for display purposes
#   category    -> groups related parameters together. This is our stand-in
#                  for "clustering" from the proposal — instead of an ML
#                  model discovering groups, we define them explicitly based
#                  on domain knowledge (which is honestly more reliable for
#                  a small number of well-understood parameters anyway).
#   confidence  -> how sure we are this threshold is the real SNI number.
#                  "verified" = confirmed via research citing SNI 2973:2011
#                  "placeholder" = representative estimate, swap out if you
#                  get access to the real BSN standard document or committee
#                  data. Being upfront about this in the demo video is a
#                  strength, not a weakness — it shows you know the
#                  difference between verified and assumed data.
#   reason_fail -> template string shown to the user when this check fails.
#                  {value} and {limit} get filled in automatically.

THRESHOLDS = {
    "moisture": {
        "limit": 5.0,
        "comparison": "max",
        "unit": "%",
        "category": "Moisture Control",
        "confidence": "verified",
        "reason_fail": "Moisture too high: {value}% exceeds the SNI 2973:2011 maximum of {limit}%",
    },
    "protein": {
        "limit": 5.0,
        "comparison": "min",
        "unit": "%",
        "category": "Nutritional Content",
        "confidence": "verified",
        "reason_fail": "Protein too low: {value}% is below the SNI 2973:2011 minimum of {limit}%",
    },
    "ash": {
        "limit": 1.0,
        "comparison": "max",
        "unit": "%",
        "category": "Mineral / Ash Content",
        "confidence": "verified",
        "reason_fail": "Ash content too high: {value}% exceeds the SNI 2973:2011 maximum of {limit}%",
    },
    "fat": {
        "limit": 30.0,
        "comparison": "max",
        "unit": "%",
        "category": "Nutritional Content",
        "confidence": "placeholder",
        "reason_fail": "Fat content too high: {value}% exceeds the reference maximum of {limit}%",
    },
    "microbial_count": {
        "limit": 10000.0,
        "comparison": "max",
        "unit": "CFU/g",
        "category": "Microbial Contamination",
        "confidence": "placeholder",
        "reason_fail": "Microbial count too high: {value} CFU/g exceeds the reference maximum of {limit} CFU/g",
    },
    "heavy_metals": {
        "limit": 0.5,
        "comparison": "max",
        "unit": "ppm",
        "category": "Contaminant / Heavy Metal",
        "confidence": "placeholder",
        "reason_fail": "Heavy metal content too high: {value} ppm exceeds the reference maximum of {limit} ppm",
    },
}


def evaluate_sample(sample: dict) -> dict:
    """
    Takes a dict of lab parameter values, e.g.:
        {"moisture": 6.2, "protein": 6.0, "ash": 0.8}

    Only parameters present in `sample` are checked — you don't have to
    send all six every time. This matters for the Predictive Simulator UI,
    where a user might want to test "what if only moisture changes?"

    Returns a dict:
        {
            "passed": bool,                # overall pass/fail
            "checked_parameters": [...],    # every param that was evaluated
            "failures": [                   # only the ones that failed
                {
                    "parameter": "moisture",
                    "value": 6.2,
                    "limit": 5.0,
                    "category": "Moisture Control",
                    "confidence": "verified",
                    "reason": "Moisture too high: 6.2% exceeds ..."
                },
                ...
            ],
            "failure_categories": [...]     # unique categories that failed,
                                             # e.g. ["Moisture Control"]
        }
    """
    failures = []
    checked_parameters = []

    for param_name, value in sample.items():
        rule = THRESHOLDS.get(param_name)

        # Skip anything we don't have a rule for, rather than crashing —
        # frontend might send extra fields we don't care about yet.
        if rule is None:
            continue

        checked_parameters.append(param_name)

        failed = False
        if rule["comparison"] == "max" and value > rule["limit"]:
            failed = True
        elif rule["comparison"] == "min" and value < rule["limit"]:
            failed = True

        if failed:
            failures.append({
                "parameter": param_name,
                "value": value,
                "limit": rule["limit"],
                "unit": rule["unit"],
                "category": rule["category"],
                "confidence": rule["confidence"],
                "reason": rule["reason_fail"].format(value=value, limit=rule["limit"]),
            })

    # dict.fromkeys() preserves order while removing duplicates — e.g. if
    # both "fat" and "protein" fail, they'd both map to "Nutritional
    # Content", and we only want that category listed once.
    failure_categories = list(dict.fromkeys(f["category"] for f in failures))

    return {
        "passed": len(failures) == 0,
        "checked_parameters": checked_parameters,
        "failures": failures,
        "failure_categories": failure_categories,
    }
