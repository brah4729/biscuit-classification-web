"""
synthetic_data.py — Generates synthetic biscuit lab samples for the
clustering demo.

IMPORTANT — say this in the video, don't hide it:
This data is NOT real production data. No committee or real lab dataset
was available for this case. These samples exist to demonstrate that the
clustering pipeline works correctly — they do not represent any real
biscuit factory's actual failure patterns. On real historical batch data,
the same clustering code would surface genuine, actionable patterns.

Design of the generator:
- "Good" samples are drawn from a normal distribution centered safely
  inside each threshold from rules.THRESHOLDS.
- A fraction of samples are deliberately pushed into one of a few
  realistic FAILURE ARCHETYPES (not just random noise), so the resulting
  clusters have a legible story: "these clusters roughly correspond to
  moisture failures, protein failures, and contamination failures."
- Moisture and microbial count are mildly correlated in the "contamination"
  archetype, matching the note in the team's own UI reference mockup:
  "High moisture metrics show a strong mathematical correlation to
  microbial growth patterns." We're intentionally reproducing that
  as a designed correlation, not discovering it from real data.
"""

import random

from rules import THRESHOLDS


def _good_value(param: str, rng: random.Random) -> float:
    """A safely-passing value for one parameter, with small random noise."""
    rule = THRESHOLDS[param]
    limit = rule["limit"]
    if rule["comparison"] == "max":
        # comfortably below the max, e.g. limit=5.0 -> center ~3.5
        center = limit * 0.7
        spread = limit * 0.12
    else:  # "min"
        # comfortably above the min, e.g. limit=5.0 -> center ~6.5
        center = limit * 1.3
        spread = limit * 0.12
    return max(0.0, rng.gauss(center, spread))


def _bad_value(param: str, rng: random.Random) -> float:
    """A value that fails this parameter's threshold, with noise."""
    rule = THRESHOLDS[param]
    limit = rule["limit"]
    if rule["comparison"] == "max":
        center = limit * 1.3  # comfortably over the max
        spread = limit * 0.15
    else:
        center = limit * 0.6  # comfortably under the min
        spread = limit * 0.15
    return max(0.0, rng.gauss(center, spread))


# Failure archetypes: named, realistic combinations of what goes wrong
# together, rather than every parameter failing independently at random.
ARCHETYPES = {
    "moisture_only": ["moisture"],
    "protein_only": ["protein"],
    "contamination": ["microbial_count", "heavy_metals"],
    "moisture_and_microbial": ["moisture", "microbial_count"],  # the correlated pair
}


def generate_synthetic_batch(n: int = 300, fail_ratio: float = 0.35, seed: int = 42) -> list[dict]:
    """
    Generate `n` synthetic samples. Roughly `fail_ratio` of them will fail
    at least one parameter, split across the archetypes above. The rest
    pass everything.

    Returns a list of dicts, each e.g.:
        {"moisture": 3.9, "protein": 6.4, "ash": 0.6, "fat": 21.0,
         "microbial_count": 850.0, "heavy_metals": 0.08}
    """
    rng = random.Random(seed)
    params = list(THRESHOLDS.keys())
    samples = []

    n_fail = int(n * fail_ratio)
    archetype_names = list(ARCHETYPES.keys())

    for i in range(n):
        sample = {p: _good_value(p, rng) for p in params}

        if i < n_fail:
            archetype = archetype_names[i % len(archetype_names)]
            for p in ARCHETYPES[archetype]:
                sample[p] = _bad_value(p, rng)

        samples.append({k: round(v, 3) for k, v in sample.items()})

    rng.shuffle(samples)
    return samples
