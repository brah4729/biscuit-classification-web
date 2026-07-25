"""
generate_dataset.py — Runs continuously, generating synthetic lab batches
and appending them to a CSV file, until you stop it with Ctrl+C.

Why a separate script instead of just raising the `n` limit on
GET /clusters/export:
A single huge HTTP request risks timing out (browser, Flask dev server,
or both) and gives you no visibility into progress while it's running.
This script writes to disk incrementally, so you can stop at any size
you want and never lose what's already been generated.

Usage:
    cd backend
    uv run python generate_dataset.py

    # optional flags:
    uv run python generate_dataset.py --batch-size 500 --output my_data.csv

Stop anytime with Ctrl+C — it finishes writing the current batch cleanly,
then prints a summary and runs clustering + an Excel export on everything
generated so far.
"""

import argparse
import csv
import os
import signal
import sys
import time

from synthetic_data import generate_synthetic_batch
from clustering import cluster_failures
from export import result_to_xlsx_bytes
from rules import THRESHOLDS


class StopRequested(Exception):
    """Raised by the SIGTERM handler below so both Ctrl+C (SIGINT, which
    Python already turns into KeyboardInterrupt automatically) and any
    other stop signal (SIGTERM — e.g. a task manager kill, an IDE's stop
    button) end up going through the exact same clean-shutdown path."""
    pass


def _handle_sigterm(signum, frame):
    raise StopRequested()


signal.signal(signal.SIGTERM, _handle_sigterm)


def main():
    parser = argparse.ArgumentParser(description="Continuously generate synthetic biscuit lab data until stopped.")
    parser.add_argument("--batch-size", type=int, default=500, help="How many samples to generate per loop iteration (default 500)")
    parser.add_argument("--output", type=str, default="synthetic_dataset.csv", help="CSV file to write to (default synthetic_dataset.csv)")
    parser.add_argument("--fail-ratio", type=float, default=0.35, help="Roughly what fraction of samples should fail (default 0.35)")
    args = parser.parse_args()

    fieldnames = list(THRESHOLDS.keys())
    output_path = os.path.join(os.path.dirname(__file__), args.output)

    file_exists = os.path.exists(output_path)
    total_written = 0
    iteration = 0

    print(f"Writing to: {output_path}")
    print("Press Ctrl+C to stop at any time — progress is saved continuously, nothing is lost.\n")

    try:
        with open(output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(output_path) == 0:
                writer.writeheader()

            while True:
                iteration += 1
                # Different seed each iteration, otherwise generate_synthetic_batch
                # would produce the exact same rows every time (it's
                # deterministic by design for reproducibility elsewhere).
                seed = int(time.time() * 1000) % 1_000_000 + iteration
                batch = generate_synthetic_batch(n=args.batch_size, fail_ratio=args.fail_ratio, seed=seed)

                for row in batch:
                    writer.writerow(row)
                f.flush()  # make sure it's actually on disk, not just buffered

                total_written += len(batch)
                print(f"\rTotal samples written: {total_written:,}", end="", flush=True)

    except (KeyboardInterrupt, StopRequested):
        print(f"\n\nStopped. {total_written:,} total samples written to {output_path}")
        _finalize(output_path, fieldnames)


def _finalize(csv_path: str, fieldnames: list[str]):
    """After stopping, read back everything generated, run clustering on
    the full accumulated dataset, and produce a ready-to-open Excel file
    — so you don't have to manually re-run clustering afterward."""
    print("Running clustering on the full dataset...")

    samples = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({k: float(v) for k, v in row.items()})

    if len(samples) < 3:
        print("Not enough samples to cluster (need at least a few). Skipping export.")
        return

    result = cluster_failures(samples, n_clusters=3)
    result["data_source"] = "synthetic"

    if not result["clusters"]:
        print(f"Warning: {result.get('warning', 'no failed samples to cluster')}")
        return

    xlsx_bytes = result_to_xlsx_bytes(result)
    xlsx_path = csv_path.replace(".csv", "_clustered.xlsx")
    with open(xlsx_path, "wb") as f:
        f.write(xlsx_bytes)

    print(f"Done. {result['failed_samples']:,} of {result['total_samples']:,} samples failed and were clustered.")
    print(f"Excel report saved to: {xlsx_path}")


if __name__ == "__main__":
    main()
