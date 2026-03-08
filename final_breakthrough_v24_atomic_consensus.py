import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path('/Users/liqi/Desktop/pet_image-main')
BASE_PATH = ROOT / 'submission_final_team_v10.csv'
REF_GLOB = 'submission_final_team_v*.csv'

OUT_TOP1 = ROOT / 'submission_final_team_v24_atomic_top1.csv'
OUT_TOP2 = ROOT / 'submission_final_team_v24_atomic_top2.csv'
MANIFEST = ROOT / 'v24_atomic_consensus_manifest.json'


def load_versions(base: pd.DataFrame):
    versions = []
    for p in sorted(ROOT.glob(REF_GLOB)):
        name = p.stem.replace('submission_final_team_', '')
        if name == 'v10':
            continue
        df = pd.read_csv(p)
        if list(df['id']) != list(base['id']):
            continue
        metrics_path = ROOT / f'{name}_oof_metrics.json'
        oof = 0.90
        if metrics_path.exists():
            try:
                oof = float(json.load(open(metrics_path)).get('stack_oof_acc', 0.90))
            except Exception:
                oof = 0.90
        drift = float((df['label'] != base['label']).mean())
        bonus = 0.03 if name in {'v15', 'v16', 'v20', 'v21_safe', 'v23_consensus_anchor', 'v23_ultra_consensus'} else 0.0
        weight = max(0.05, (oof - 0.86)) * math.exp(-3.2 * drift) + bonus
        versions.append({'name': name, 'df': df, 'weight': float(weight), 'oof': oof, 'drift': drift})
    return versions


def find_consensus_flips(base: pd.DataFrame, versions):
    flips = []
    for i, row in base.iterrows():
        weighted_votes = {}
        raw_labels = []
        total_w = 0.0
        for v in versions:
            lab = v['df'].iloc[i]['label']
            raw_labels.append((v['name'], lab, v['weight']))
            weighted_votes[lab] = weighted_votes.get(lab, 0.0) + v['weight']
            total_w += v['weight']

        top_label, top_weight = sorted(weighted_votes.items(), key=lambda kv: kv[1], reverse=True)[0]
        support = top_weight / max(total_w, 1e-9)
        agree = sum(1 for _, lab, _ in raw_labels if lab == top_label)

        if top_label != row['label'] and support >= 0.60 and agree >= 3:
            flips.append({
                'id': row['id'],
                'old': row['label'],
                'new': top_label,
                'support': round(float(support), 6),
                'agree_count': int(agree),
                'weighted_votes': {k: round(float(v), 6) for k, v in weighted_votes.items()},
            })

    flips.sort(key=lambda x: (x['support'], x['agree_count']), reverse=True)
    return flips


def apply_topk(base: pd.DataFrame, flips, k: int):
    out = base.copy()
    keep = flips[:k]
    mp = {x['id']: x['new'] for x in keep}
    out['label'] = [mp.get(i, l) for i, l in zip(out['id'], out['label'])]
    return out, keep


def main():
    base = pd.read_csv(BASE_PATH)
    versions = load_versions(base)
    flips = find_consensus_flips(base, versions)

    top1_df, top1 = apply_topk(base, flips, 1)
    top2_df, top2 = apply_topk(base, flips, 2)

    top1_df.to_csv(OUT_TOP1, index=False)
    top2_df.to_csv(OUT_TOP2, index=False)

    manifest = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'base': str(BASE_PATH),
        'version_count': len(versions),
        'versions': [
            {'name': v['name'], 'weight': v['weight'], 'oof': v['oof'], 'drift_vs_v10': v['drift']}
            for v in versions
        ],
        'selection_rule': 'top_label!=v10 AND weighted_support>=0.60 AND agree_count>=3',
        'all_candidate_flips': flips,
        'top1_flips': top1,
        'top2_flips': top2,
        'outputs': {
            'top1': str(OUT_TOP1),
            'top2': str(OUT_TOP2),
        },
    }
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    print(f'Generated: {OUT_TOP1}')
    print(f'Generated: {OUT_TOP2}')
    print(f'Generated: {MANIFEST}')
    print(f'candidate_flips={len(flips)} top1={len(top1)} top2={len(top2)}')


if __name__ == '__main__':
    main()
