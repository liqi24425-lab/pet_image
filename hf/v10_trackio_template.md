# v10 Trackio Logging Contract

If `trackio` is available, each run logs:

- `stack_oof_acc`
- `stack_oof_macro_f1`
- `dino_oof_acc`
- `effnet_oof_acc`
- `clip_oof_acc`
- `pseudo_count`
- `pseudo_ratio`
- `fallback_triggered`

If `trackio` is unavailable, the run remains valid and local JSON artifacts are the source of truth.
