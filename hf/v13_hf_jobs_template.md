# v13 HF Jobs Template (analysis-only)

Use this to run `v13` remotely on Hugging Face Jobs for offline ablations.

```python
hf_jobs("uv", {
  "command": "python final_breakthrough_v13_hybrid_stack.py",
  "cwd": "/workspace/pet_image-main",
  "flavor": "a10g-large",
  "secrets": {"HF_TOKEN": "$HF_TOKEN"},
  "timeout": 21600,
  "env": {
    "V13_MODE": "analysis",
    "V13_TRACKIO": "1"
  }
})
```

Required outputs:
- `v13_oof_metrics.json`
- `v13_run_manifest.json`
- `submission_final_team_v13.csv` (promote only after local replay)
