# v12 HF Jobs Template (analysis-only)

Purpose: run large ablation jobs remotely while keeping local `submission_final_team_v12.csv` as the only promotion target.

```python
hf_jobs("uv", {
  "command": "python final_breakthrough_v12_effective_hf_fusion.py",
  "cwd": "/workspace/pet_image-main",
  "flavor": "a10g-large",
  "secrets": {"HF_TOKEN": "$HF_TOKEN"},
  "timeout": 21600,
  "env": {
    "V12_MODE": "analysis",
    "V12_TRACKIO": "1"
  }
})
```

Required artifacts:
- `v12_oof_metrics.json`
- `v12_run_manifest.json`
- `submission_final_team_v12.csv` (analysis-only until local replay)
