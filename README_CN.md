# STA314H 宠物面部表情分类（中文说明）

当前公开榜最佳记录：`0.92666`（历史已提交版本）

## 1. 当前项目目标

- 任务：3 分类（`Angry / Happy / Sad`）
- 数据规模：训练 450，测试 300
- 团队策略：每天提交次数有限，采用“单版本理性提交”

## 2. 最近版本结论（你现在最关心）

- `v10`：已验证可用，曾拿到更高公开榜分数。
- `v12`：离线无稳定收益，不建议提交。
- `v13`：离线看起来更好，但线上公开榜退步（`0.92000`），暂不作为默认提交版本。
- `v14`：离线回退（stack OOF 下降到 `0.9000`），淘汰。
- `v15`：离线显著提升且与 v10 漂移很小，作为下一次“单次尝试”候选。

`v13` 离线核心结果（`v13_oof_metrics.json`）：

- `stack_oof_acc = 0.9067`
- `stack_oof_macro_f1 = 0.9066`
- `weighted_oof_acc = 0.9067`
- `meta_true_oof_acc = 0.8956`
- `hybrid_alpha = 0.25`

解释：
- 本轮把最终决策头从“直接 meta”改成“`hybrid(meta + weighted)`”，并且用真实 OOF 校准，降低了小样本下的 stack 不稳定问题。

## 3. 为什么撤掉 EffNetV2-L

在 `v11` 中，EffNetV2-L 的 OOF 大幅退化到约 `0.473`，会拖累整体融合质量。  
因此回滚到 `EffNet-B5`，这是当前数据规模下更稳定的 CNN 分支。

## 4. 你现在该提交哪个文件

当前默认提交文件：

- `submission_final_team_v10.csv`

说明：
- `v13` 已在 2026-03-06 验证公开榜 `0.92000`，低于 `v10` 的 `0.92666`，因此归档为 `analysis-only`。
- `v15` 是下一次唯一可尝试候选：`submission_final_team_v15.csv`（只尝试一次）。

## 5. 关键脚本与产物

- `v13` 主脚本：`final_breakthrough_v13_hybrid_stack.py`
- `v14` 主脚本：`final_breakthrough_v14_robust_prior.py`
- `v15` 主脚本：`final_breakthrough_v15_seed_ensemble.py`
- `v15` 指标：`v15_oof_metrics.json`
- `v15` 运行配置：`v15_run_manifest.json`

## 6. 复现命令

```bash
source .venv/bin/activate
python final_breakthrough_v15_seed_ensemble.py
```

生成：

- `submission_final_team_v15.csv`
- `v15_oof_metrics.json`
- `v15_run_manifest.json`

## 7. 团队协作建议（简版）

1. 每轮只允许一个“可提交文件”进入群里。
2. 提交前先看 OOF：若不高于当前主线，不提交。
3. 保留变更日志（本轮已写入 README / README_CN）。
