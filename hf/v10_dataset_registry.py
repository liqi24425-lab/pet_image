import csv
import hashlib
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


def append_from_metrics(metrics_path: str = "v10_oof_metrics.json", manifest_path: str = "v10_run_manifest.json") -> None:
    ensure_registry(REG_PATH)
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

    with open(REG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writerow(row)

    print(f"Appended registry row for run_id={row['run_id']}")


if __name__ == "__main__":
    append_from_metrics()
