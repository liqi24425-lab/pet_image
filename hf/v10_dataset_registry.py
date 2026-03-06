import argparse
import csv
import json
import os
from datetime import datetime, timezone

REG_PATH = "hf/v10_experiments_registry.csv"

HEADER = [
    "timestamp_utc",
    "run_id",
    "config_hash",
    "stack_oof_acc",
    "stack_oof_macro_f1",
    "pseudo_count",
    "pseudo_ratio",
    "fallback_triggered",
    "artifact_submission",
    "artifact_metrics",
    "artifact_manifest",
]


def ensure_registry(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)


def append_from_metrics(
    metrics_path: str = "v10_oof_metrics.json",
    manifest_path: str = "v10_run_manifest.json",
    registry_path: str = REG_PATH,
) -> None:
    ensure_registry(registry_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": metrics.get("run_id", "unknown"),
        "config_hash": metrics.get("config_hash", "unknown"),
        "stack_oof_acc": metrics.get("stack_oof_acc"),
        "stack_oof_macro_f1": metrics.get("stack_oof_macro_f1"),
        "pseudo_count": metrics.get("pseudo_count"),
        "pseudo_ratio": metrics.get("pseudo_ratio"),
        "fallback_triggered": metrics.get("fallback_triggered"),
        "artifact_submission": manifest.get("outputs", {}).get("final_submission", ""),
        "artifact_metrics": manifest.get("outputs", {}).get("metrics", ""),
        "artifact_manifest": "v10_run_manifest.json",
    }

    with open(registry_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writerow(row)

    print(f"Appended registry row for run_id={row['run_id']} -> {registry_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append experiment metrics to CSV registry.")
    parser.add_argument("--metrics", default="v10_oof_metrics.json", help="Path to metrics JSON")
    parser.add_argument("--manifest", default="v10_run_manifest.json", help="Path to manifest JSON")
    parser.add_argument("--registry", default=REG_PATH, help="Path to registry CSV")
    args = parser.parse_args()
    append_from_metrics(metrics_path=args.metrics, manifest_path=args.manifest, registry_path=args.registry)
