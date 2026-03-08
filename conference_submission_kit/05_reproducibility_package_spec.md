# 05. Reproducibility Package Specification

## Required top-level assets
1. `README_reproduce.md`
2. `requirements.txt` (version-locked)
3. `configs/` (all experiment configs)
4. `scripts/` (train/eval/export)
5. `artifacts_index.csv`
6. `results/` (tables and figure source data)

## Required per experiment
For each final experiment row in paper tables, provide:
1. Script path.
2. Config hash.
3. Random seed.
4. Start/end timestamp.
5. Metrics json path.
6. Manifest json path.
7. Output file path.

## Recommended artifact schema
- `results/<exp_id>/metrics.json`
- `results/<exp_id>/manifest.json`
- `results/<exp_id>/predictions.csv`
- `results/<exp_id>/log.txt`

## Reproduction verification protocol
1. Run from clean virtual environment.
2. Execute listed commands in order.
3. Compare generated metrics with reported paper numbers.
4. Allow tolerance for hardware-related nondeterminism (predefined threshold).

## Quality gates
- [ ] Every table number is backed by a concrete metrics file.
- [ ] Every figure can be regenerated from source data.
- [ ] Missing files fail CI check.

## Suggested automation
1. Add `make reproduce-main` for main results.
2. Add `make verify-artifacts` to ensure all referenced files exist.
3. Add `make build-tables` to regenerate CSV/LaTeX tables.
