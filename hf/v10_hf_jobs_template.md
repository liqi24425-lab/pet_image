# v10 Hugging Face Jobs Template (analysis-only)

Use this template for cloud/offline expansion experiments. Do not submit cloud outputs directly; only use them to recommend configs for local `final_breakthrough_v10.py`.

## Example payload

```python
hf_jobs("uv", {
  "command": "python final_breakthrough_v10.py",
  "cwd": "/workspace/pet_image-main",
  "flavor": "a10g-small",
  "secrets": {"HF_TOKEN": "$HF_TOKEN"},
  "timeout": 14400,
  "env": {
    "V10_MODE": "analysis",
    "V10_TRACKIO": "1"
  }
})
```

## Required outputs per cloud run

- `v10_oof_metrics.json`
- `v10_run_manifest.json`
- `submission_final_team_v10.csv` (analysis-only copy)

## Promotion rule

Only promote a config to team submission if local rerun on current repo reproduces expected behavior and passes drift gates.
