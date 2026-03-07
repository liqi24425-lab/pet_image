# v22 文献来源与实现映射（必须可追溯）

本文件记录 `final_breakthrough_v22_lit_noise_consistency.py` 的两条核心改动及其文献来源。

## 1) 噪声鲁棒训练（Noise-aware Reweight）

- 文献来源：
  - LA-Net: [https://arxiv.org/abs/2307.09023](https://arxiv.org/abs/2307.09023)
- 借鉴思想：
  - 对疑似噪声样本降权，避免模型过拟合错误标签。
- 在 v22 的实现：
  - 函数 `compute_noise_reweight(...)`
  - 先基于三骨干 OOF 概率计算置信度/熵，再生成训练样本权重并重训一次 probe。

## 2) 多视角一致性门控（Multi-view Consistency Gate）

- 文献来源：
  - Multi-Task Multi-Modal Self-Supervised Learning for FER: [https://arxiv.org/abs/2404.10904](https://arxiv.org/abs/2404.10904)
- 借鉴思想：
  - 用多视角/多分支一致性约束提高鲁棒性，减少单视角偶然误判。
- 在 v22 的实现：
  - 函数 `gate_tta_probs(...)`
  - 将测试视角拆为 `global` 与 `roi` 两组；冲突时按置信差门控，否则融合。

## 3) 代码入口与产物

- 代码：
  - `final_breakthrough_v22_lit_noise_consistency.py`
- 默认输出：
  - `submission_final_team_v22.csv`
  - `v22_oof_metrics.json`
  - `v22_run_manifest.json`

