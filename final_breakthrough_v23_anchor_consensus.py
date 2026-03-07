import json
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

ROOT = "/Users/liqi/Desktop/pet_image-main"

BASE = f"{ROOT}/submission_final_team_v10.csv"
REFS = {
    "v15": f"{ROOT}/submission_final_team_v15.csv",
    "v16": f"{ROOT}/submission_final_team_v16.csv",
    "v20": f"{ROOT}/submission_final_team_v20.csv",
    "v22": f"{ROOT}/submission_final_team_v22.csv",
}

OUT_STRONG = f"{ROOT}/submission_final_team_v23_consensus_anchor.csv"
OUT_ULTRA = f"{ROOT}/submission_final_team_v23_ultra_consensus.csv"
MANIFEST = f"{ROOT}/v23_consensus_manifest.json"


def build_candidate(base: pd.DataFrame, ref_dfs: dict, min_votes: int):
    out = base.copy()
    changes = []
    for i, row in base.iterrows():
        labels = [df.iloc[i]["label"] for df in ref_dfs.values()]
        top, count = Counter(labels).most_common(1)[0]
        if count >= min_votes and top != row["label"]:
            out.at[i, "label"] = top
            changes.append({
                "id": row["id"],
                "old": row["label"],
                "new": top,
                "votes": int(count),
                "ref_labels": labels,
            })
    return out, changes


def main():
    base = pd.read_csv(BASE)
    ref_dfs = {k: pd.read_csv(v) for k, v in REFS.items()}

    for df in ref_dfs.values():
        if list(df["id"]) != list(base["id"]):
            raise ValueError("ID order mismatch between base and reference submissions")

    strong_df, strong_changes = build_candidate(base, ref_dfs, min_votes=3)
    ultra_df, ultra_changes = build_candidate(base, ref_dfs, min_votes=4)

    strong_df.to_csv(OUT_STRONG, index=False)
    ultra_df.to_csv(OUT_ULTRA, index=False)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base": BASE,
        "references": REFS,
        "rules": {
            "strong": "apply flip if >=3/4 references agree and differ from base",
            "ultra": "apply flip if 4/4 references agree and differ from base",
        },
        "outputs": {
            "strong": OUT_STRONG,
            "ultra": OUT_ULTRA,
        },
        "strong_change_count": len(strong_changes),
        "ultra_change_count": len(ultra_changes),
        "strong_changes": strong_changes,
        "ultra_changes": ultra_changes,
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    print(f"Generated: {OUT_STRONG}")
    print(f"Generated: {OUT_ULTRA}")
    print(f"Generated: {MANIFEST}")
    print(f"strong_changes={len(strong_changes)}, ultra_changes={len(ultra_changes)}")


if __name__ == "__main__":
    main()
