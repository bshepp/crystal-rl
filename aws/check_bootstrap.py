#!/usr/bin/env python3
"""Monitor a Phase-1 AWS bootstrap run and retrieve its S3 results.

Historical context: this script targets a specific EC2 instance
(`INSTANCE_ID` below) that ran the original 794-point bootstrap DFT job in
February 2026 (see ROADMAP Phase 1). That instance has long since self-
terminated; the script's status-check path is now mostly useful as a
template for monitoring future bootstrap-style runs.

The `--download` and `--log` paths still work against the S3 bucket and
remain useful for pulling historical bootstrap artifacts.

To reuse for a new run, edit `INSTANCE_ID` (and optionally `BUCKET`).

Usage:
    python aws/check_bootstrap.py              # Check status
    python aws/check_bootstrap.py --download   # Download results from S3
    python aws/check_bootstrap.py --log        # View the run log
"""

import argparse
import json
import os
import subprocess
import sys

# Original Phase-1 bootstrap instance (self-terminated Feb 2026, kept for record).
INSTANCE_ID = "i-0d3ff917412bd0f51"
BUCKET = "rl-materials-bootstrap-290318879194"
REGION = "us-east-1"


def run_aws(cmd: list[str]) -> str:
    result = subprocess.run(
        ["aws"] + cmd + ["--region", REGION],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"AWS error: {result.stderr}", file=sys.stderr)
    return result.stdout.strip()


def check_status():
    """Check instance state and S3 results."""
    print("=== Instance Status ===")
    out = run_aws([
        "ec2", "describe-instances",
        "--instance-ids", INSTANCE_ID,
        "--query", "Reservations[0].Instances[0].{State:State.Name,Type:InstanceType,LaunchTime:LaunchTime}",
        "--output", "json",
    ])
    if out:
        info = json.loads(out)
        print(f"  State:  {info['State']}")
        print(f"  Type:   {info['Type']}")
        print(f"  Launch: {info['LaunchTime']}")

    print("\n=== S3 Results ===")
    out = run_aws(["s3", "ls", f"s3://{BUCKET}/results/bootstrap/", "--recursive"])
    if out:
        for line in out.strip().split("\n"):
            print(f"  {line}")
    else:
        print("  (no results yet)")

    print("\n=== S3 Log ===")
    out = run_aws(["s3", "ls", f"s3://{BUCKET}/logs/"])
    if out:
        print(f"  {out}")
    else:
        print("  (no log yet)")


def download_results():
    """Download all results from S3 to local data/ directory."""
    print("Downloading results...")
    os.makedirs("data/bootstrap", exist_ok=True)
    run_aws(["s3", "sync", f"s3://{BUCKET}/results/bootstrap/", "data/bootstrap/"])
    print("Done. Check data/bootstrap/ for results.")


def view_log():
    """Download and print the run log."""
    out = run_aws(["s3", "cp", f"s3://{BUCKET}/logs/bootstrap-run.log", "-"])
    if out:
        print(out)
    else:
        print("(log not available yet)")


def main():
    parser = argparse.ArgumentParser(description="Monitor AWS bootstrap")
    parser.add_argument("--download", action="store_true", help="Download results")
    parser.add_argument("--log", action="store_true", help="View run log")
    args = parser.parse_args()

    if args.download:
        download_results()
    elif args.log:
        view_log()
    else:
        check_status()


if __name__ == "__main__":
    main()
