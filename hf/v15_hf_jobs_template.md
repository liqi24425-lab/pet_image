# v15 HF Jobs Template (analysis-only)

Use this template to run offline ablations for `v15` on Hugging Face Jobs.

```python
hf_jobs("uv", {
  "command": "python final_breakthrough_v15_seed_ensemble.py",
  "cwd": "/workspace/pet_image-main",
  "flavor": "a10g-large",
  "secrets": {"HF_TOKEN": "$HF_TOKEN"},
  "timeout": 21600,
  "env": {
    "V15_MODE": "analysis",
    "V15_TRACKIO": "1"
  }
})
```

Required outputs:
- `v15_oof_metrics.json`
- `v15_run_manifest.json`
- `submission_final_team_v15.csv` (promote only after local + leaderboard gate)
