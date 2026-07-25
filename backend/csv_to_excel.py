"""
csv_to_excel.py — Converts an existing synthetic_dataset.csv (or any CSV
with the same columns) into a readable Excel report, without regenerating
any data. Useful if you already have a CSV from generate_dataset.py and
just want the Excel version.

Usage:
    cd backend
    uv run python csv_to_excel.py synthetic_dataset.csv
    uv run python csv_to_excel.py synthetic_dataset.csv --n-clusters 4
"""

import argparse
import csv

from clustering import cluster_failures
from export import result_to_xlsx_bytes
from rules import THRESHOLDS


def main():
    parser = argparse.ArgumentParser(description="Convert a lab-data CSV into a clustered Excel report.")
    parser.add_argument("csv_path", help="Path to the CSV file to convert")
    parser.add_argument("--n-clusters", type=int, default=3, help="How many clusters to form (default 3)")
    parser.add_argument("--output", type=str, default=None, help="Output .xlsx path (default: same name as input, with _clustered.xlsx)")
    args = parser.parse_args()

    print(f"Reading: {args.csv_path}")
    samples = []
    with open(args.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only keep columns we actually have rules for — ignores any
            # extra columns (e.g. cluster_id, failure_categories) if you
            # accidentally point this at an already-exported file.
            sample = {k: float(v) for k, v in row.items() if k in THRESHOLDS}
            samples.append(sample)

    print(f"Loaded {len(samples):,} samples")

    if len(samples) < args.n_clusters:
        print(f"Not enough samples ({len(samples)}) to form {args.n_clusters} clusters. Try fewer clusters or a bigger dataset.")
        return

    result = cluster_failures(samples, n_clusters=args.n_clusters)
    result["data_source"] = "synthetic"

    if not result["clusters"]:
        print(f"Warning: {result.get('warning', 'no failed samples to cluster')}")
        return

    output_path = args.output or args.csv_path.rsplit(".", 1)[0] + "_clustered.xlsx"
    xlsx_bytes = result_to_xlsx_bytes(result)
    with open(output_path, "wb") as f:
        f.write(xlsx_bytes)

    print(f"{result['failed_samples']:,} of {result['total_samples']:,} samples failed and were clustered.")
    print(f"Excel report saved to: {output_path}")


if __name__ == "__main__":
    main()
