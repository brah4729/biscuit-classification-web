"""
clustering.py — Groups FAILED samples into clusters using KMeans, to
satisfy the brief's "identifying product quality failure patterns" /
"grouping samples" requirement.

Why cluster only the failures, not all samples:
The brief's clustering ask is about understanding failure patterns
specifically ("identify product quality failure patterns"). Passing
samples don't have a "pattern" worth discovering — they just passed.
Clustering only failures keeps the result legible: each cluster answers
"what kind of failure is this?" rather than mixing pass/fail into
meaningless geometric groups.

Why KMeans specifically: see the team discussion in AGENTS.md. Short
version — our failure types are a small, roughly-known number of
categories (moisture, protein, contamination, combined), which is exactly
the case KMeans handles well and DBSCAN/hierarchical would be overkill for.
"""

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from rules import evaluate_sample


def cluster_failures(samples: list[dict], n_clusters: int = 3, seed: int = 42) -> dict:
    """
    samples: list of raw lab-parameter dicts (e.g. from synthetic_data.py)

    Steps:
    1. Run every sample through the existing rule engine (rules.py) —
       we reuse it rather than re-implementing pass/fail logic here.
    2. Keep only the ones that failed.
    3. Standardize their parameter values (StandardScaler) — this matters
       because our parameters are on wildly different scales (e.g.
       microbial_count in the thousands, heavy_metals in fractions of a
       ppm). Without scaling, KMeans would basically only "see" whichever
       parameter has the biggest raw numbers.
    4. Run KMeans to group them.
    5. For each cluster, summarize which failure category dominates it,
       so the result is human-readable, not just cluster numbers.

    Returns:
        {
            "total_samples": int,
            "failed_samples": int,
            "clusters": [
                {
                    "cluster_id": 0,
                    "size": 24,
                    "dominant_category": "Moisture Control",
                    "mean_values": {"moisture": 6.8, "protein": 6.1, ...}
                },
                ...
            ],
            "points": [
                {"cluster_id": 0, "categories": [...], "values": {...}},
                ...
            ]
        }
    """
    failed = []
    for sample in samples:
        result = evaluate_sample(sample)
        if not result["passed"]:
            failed.append({"values": sample, "categories": result["failure_categories"]})

    if len(failed) < n_clusters:
        # Not enough failed samples to form the requested number of
        # clusters — fail loudly with a clear message rather than letting
        # sklearn throw a cryptic error.
        return {
            "total_samples": len(samples),
            "failed_samples": len(failed),
            "clusters": [],
            "points": [],
            "warning": f"Only {len(failed)} failed samples generated, need at least {n_clusters} to cluster. Try a larger batch size.",
        }

    # All samples share the same parameter keys (from THRESHOLDS), so we
    # can safely use the first sample's keys as our consistent feature order.
    feature_names = list(failed[0]["values"].keys())
    X = np.array([[s["values"][f] for f in feature_names] for s in failed])

    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    clusters = []
    points = []

    for cluster_id in range(n_clusters):
        member_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        members = [failed[i] for i in member_indices]

        # Dominant category = the most common failure category among this
        # cluster's members. This turns "cluster 0" into something like
        # "Moisture Control" that a non-technical auditor can read.
        all_categories = [cat for m in members for cat in m["categories"]]
        dominant_category = Counter(all_categories).most_common(1)[0][0] if all_categories else "Mixed"

        mean_values = {
            f: round(float(np.mean([m["values"][f] for m in members])), 3)
            for f in feature_names
        }

        clusters.append({
            "cluster_id": cluster_id,
            "size": len(members),
            "dominant_category": dominant_category,
            "mean_values": mean_values,
        })

    for i, s in enumerate(failed):
        points.append({
            "cluster_id": int(labels[i]),
            "categories": s["categories"],
            "values": s["values"],
        })

    return {
        "total_samples": len(samples),
        "failed_samples": len(failed),
        "clusters": clusters,
        "points": points,
    }
