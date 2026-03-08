# 04. Data Release and Compliance

## Core rule
Do not release private competition test labels or restricted assets unless the competition/data license explicitly allows it.

## What to check before any release
1. Competition rules on redistribution.
2. Dataset license terms (train/test splits, annotations, derived outputs).
3. Team-internal agreement on ownership.
4. Institution policy (if applicable).

## Recommended release strategy

### Option A (preferred): release reproducible code + logs only
Release:
1. Training/evaluation scripts.
2. Config files and seeds.
3. OOF metrics and manifests.
4. Instructions to run on licensed data.

### Option B: release a newly curated public benchmark subset
1. Collect data with explicit redistribution rights.
2. Build annotation guideline document.
3. Use at least two annotators and report inter-annotator agreement.
4. Publish split json and metadata card.

## Minimum dataset card fields
1. Data source and collection process.
2. Label definitions.
3. Known biases.
4. Recommended and prohibited uses.
5. License and citation.

## Prediction release policy
1. Never publish files that can leak private labels.
2. If sharing predictions, clarify they are model outputs only.
3. Include model version and checksum for traceability.

## Paper disclosure text (recommended)
- State that competition private test labels are not redistributed.
- State that released artifacts are code/config/manifests for reproducibility.
- If a public subset is released, cite its separate license and protocol.
