"""
export.py — Turns a clustering result (from clustering.cluster_failures)
into an actual downloadable file, so the data can be opened directly in
Excel/Sheets instead of read as raw JSON.

Two formats:
- xlsx: two sheets — "Samples" (every failed sample, its values, and
  which cluster it landed in) and "Cluster Summary" (one row per cluster
  with size, dominant category, and mean parameter values). This is the
  more useful format since it keeps both views in one file.
- csv: a single flat table (the per-sample view only) — for anyone who
  just wants something quick without opening Excel.
"""

import io

import pandas as pd


def _samples_dataframe(result: dict) -> pd.DataFrame:
    """One row per failed sample: its raw values, which cluster it's in,
    and which rule-based categories it failed on."""
    rows = []
    for point in result["points"]:
        row = dict(point["values"])
        row["cluster_id"] = point["cluster_id"]
        row["failure_categories"] = ", ".join(point["categories"])
        rows.append(row)
    return pd.DataFrame(rows)


def _cluster_summary_dataframe(result: dict) -> pd.DataFrame:
    """One row per cluster: size, dominant category, mean values."""
    rows = []
    for c in result["clusters"]:
        row = {
            "cluster_id": c["cluster_id"],
            "size": c["size"],
            "dominant_category": c["dominant_category"],
        }
        row.update({f"mean_{k}": v for k, v in c["mean_values"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def result_to_xlsx_bytes(result: dict) -> bytes:
    """Returns raw .xlsx file bytes, ready to send as a Flask response."""
    samples_df = _samples_dataframe(result)
    summary_df = _cluster_summary_dataframe(result)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Cluster Summary", index=False)
        samples_df.to_excel(writer, sheet_name="Samples", index=False)
    buffer.seek(0)
    return buffer.read()


def result_to_csv_bytes(result: dict) -> bytes:
    """Returns raw .csv bytes (per-sample view only)."""
    samples_df = _samples_dataframe(result)
    buffer = io.StringIO()
    samples_df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
