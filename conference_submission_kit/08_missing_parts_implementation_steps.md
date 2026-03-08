# 08. Missing Parts: Detailed Implementation Steps

This file provides command-level execution steps for missing publication-quality components.

## 0) Setup
1. Activate env:
```bash
source .venv/bin/activate
```
2. Create working folders:
```bash
mkdir -p research_runs/{ablation,stats,robustness,paper_tables}
```
3. Snapshot current code state:
```bash
git rev-parse HEAD > research_runs/code_commit.txt
```

## 1) Multi-seed repeat for core versions
Goal: collect stable mean±std for `v10`, `v15`, `v20`, `v22`, `v26`.

### Step 1.1 Run scripts per seed
Run each experiment 5 times (seeds: 42,52,62,72,82). If script does not expose seed env, temporarily create seed-specific config variants.

Suggested run record format:
```text
research_runs/stats/<version>_seed<seed>_metrics.json
research_runs/stats/<version>_seed<seed>_manifest.json
```

### Step 1.2 Aggregate
Create a CSV with columns:
- version, seed, oof_acc, oof_macro_f1, drift_ratio

Compute summary:
- mean, std, min, max per version.

Acceptance:
- `research_runs/paper_tables/table_main_mean_std.csv` exists.

## 2) Backbone ablation (mandatory)
Goal: prove component necessity.

### Step 2.1 Single-backbone runs
Implement or toggle runs for:
- dino-only
- effnet-only
- clip-only

### Step 2.2 Pairwise and full
- dino+effnet
- dino+clip
- effnet+clip
- dino+effnet+clip

### Step 2.3 Table generation
Output:
- `research_runs/paper_tables/table_backbone_ablation.csv`

Acceptance:
- table has all 7 rows with same protocol.

## 3) Module ablation ladder (mandatory)
Goal: justify each module by incremental gain.

Run ladder:
1. weighted base
2. +meta
3. +calibration
4. +hybrid alpha
5. +tta consistency gate
6. +noise reweight
7. +pseudo gate
8. +safety fallback

Output:
- `research_runs/paper_tables/table_module_ablation.csv`
- `research_runs/paper_tables/table_module_delta.csv`

Acceptance:
- each row has explicit delta from previous row.

## 4) Robustness analysis (mandatory)
Goal: show behavior under stress, not only aggregate accuracy.

### Step 4.1 Construct stress transforms
At inference only, apply controlled perturbation sets:
- brightness {0.7, 0.85, 1.15}
- gaussian blur {sigma 1,2}
- crop shift {left/right/up/down}

### Step 4.2 Evaluate
For each setting, record metric drop vs clean input.

Output:
- `research_runs/paper_tables/table_robustness.csv`
- `research_runs/robustness/curve_robustness.png`

Acceptance:
- robustness curve included in paper figure list.

## 5) Failure-case atlas (mandatory)
Goal: qualitative evidence.

### Step 5.1 Extract hardest samples
Select top 30 errors by confidence mismatch or entropy.

### Step 5.2 Categorize
Use fixed tags:
- background bias
- occlusion
- pose extreme
- ambiguous expression
- low quality

### Step 5.3 Assemble figure
Output:
- `research_runs/robustness/failure_atlas_v1.pdf`
- `research_runs/paper_tables/table_failure_taxonomy.csv`

Acceptance:
- at least one figure + one taxonomy table.

## 6) External validity (strongly recommended)
Goal: reduce reviewer concern on dataset-specific overfitting.

### Step 6.1 Dataset route
Choose one publicly licensable external set.

### Step 6.2 Minimal protocol
- zero-shot frozen feature transfer
- light adaptation with small labeled subset

### Step 6.3 Reporting
Output:
- `research_runs/paper_tables/table_external_validity.csv`

Fallback if no external dataset:
- include stronger controlled stress benchmark and clearly state limitation.

## 7) Statistical significance section
Goal: support claims with tests.

### Step 7.1 Paired tests
For key comparisons (`v10` vs `v22`, `v15` vs `v20`), run paired significance tests over fold-level scores.

### Step 7.2 Report
Output:
- `research_runs/paper_tables/table_significance.csv`

Acceptance:
- p-values and effect sizes present.

## 8) Reproducibility release prep
1. Create `README_reproduce.md`.
2. Lock dependencies in `requirements.txt`.
3. Build `artifacts_index.csv` linking all paper numbers to file paths.
4. Dry-run from clean env.

Acceptance:
- teammate can regenerate paper tables with commands only.

## 9) Final manuscript assembly sequence
1. Methods
2. Setup
3. Main results
4. Ablations
5. Robustness/failure analysis
6. Discussion + limitations
7. Intro + abstract

## 10) Final quality gates
- [ ] Every claim has evidence file path.
- [ ] Every table can be regenerated.
- [ ] No private-label leakage.
- [ ] Final PDF and supplementary pass formatting checks.
