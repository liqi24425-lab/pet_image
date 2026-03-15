import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B5_Weights, efficientnet_b5
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

# Optional tracking: only used if installed.
try:
    import trackio  # type: ignore
except Exception:
    trackio = None


@dataclass
class V10Config:
    seed: int = 42
    n_splits: int = 5
    batch_size: int = 16
    linear_epochs: int = 120
    linear_lr: float = 3e-3
    linear_weight_decay: float = 1e-3
    label_smoothing: float = 0.08

    pseudo_enabled: bool = True
    pseudo_weight: float = 0.30
    pseudo_conf_threshold: float = 0.92
    pseudo_entropy_threshold: float = 0.23
    pseudo_min_count: int = 6
    pseudo_max_ratio: float = 0.20

    # Runtime safety gates for "single stable submission"
    max_label_shift_ratio: float = 0.12
    min_stack_oof_acc: float = 0.88

    initial_model_weights_dino: float = 0.45
    initial_model_weights_effnet: float = 0.35
    initial_model_weights_clip: float = 0.20


CFG = V10Config()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_dirs() -> Tuple[str, str]:
    if os.path.isdir("train/train"):
        train_dir = "train/train"
    elif os.path.isdir("train"):
        train_dir = "train"
    else:
        raise FileNotFoundError("Cannot find train directory")

    if os.path.isdir("test/test"):
        test_dir = "test/test"
    elif os.path.isdir("test"):
        test_dir = "test"
    else:
        raise FileNotFoundError("Cannot find test directory")

    return train_dir, test_dir


class PathDataset(Dataset):
    def __init__(self, paths: List[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


@dataclass
class BackboneSpec:
    name: str
    model: nn.Module
    transform_map: Dict[str, transforms.Compose]


def build_transforms() -> Dict[str, Dict[str, transforms.Compose]]:
    eff_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    clip_norm = transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711],
    )

    def make_ttas(norm):
        return {
            "orig": transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm]),
            "flip": transforms.Compose(
                [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm]
            ),
            "zoom": transforms.Compose(
                [transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), norm]
            ),
            "zoom_flip": transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                    norm,
                ]
            ),
        }

    return {"effnet": make_ttas(eff_norm), "dino": make_ttas(eff_norm), "clip": make_ttas(clip_norm)}


def load_backbones(device: torch.device) -> Dict[str, BackboneSpec]:
    tmap = build_transforms()

    eff = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
    eff.classifier[1] = nn.Identity()
    eff = eff.to(device).eval()

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device).eval()
    clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    return {
        "effnet": BackboneSpec("effnet", eff, tmap["effnet"]),
        "dino": BackboneSpec("dino", dino, tmap["dino"]),
        "clip": BackboneSpec("clip", clip, tmap["clip"]),
    }


def extract_features(spec: BackboneSpec, paths: List[str], tta_key: str, device: torch.device, desc: str) -> np.ndarray:
    loader = DataLoader(PathDataset(paths, spec.transform_map[tta_key]), batch_size=CFG.batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{spec.name}-{desc}-{tta_key}"):
            batch = batch.to(device)
            if spec.name == "clip":
                out = spec.model(pixel_values=batch).image_embeds
            else:
                out = spec.model(batch)
            feats.append(out.cpu().numpy())
    return np.vstack(feats)


def class_weights_from_labels(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    device: torch.device,
    sample_weight: np.ndarray = None,
) -> nn.Module:
    model = nn.Linear(X.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.linear_lr, weight_decay=CFG.linear_weight_decay)
    cweights = class_weights_from_labels(y, num_classes).to(device)
    ce = nn.CrossEntropyLoss(weight=cweights, label_smoothing=CFG.label_smoothing, reduction="none")

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    if sample_weight is None:
        w_t = torch.ones(len(y), dtype=torch.float32, device=device)
    else:
        w_t = torch.tensor(sample_weight, dtype=torch.float32, device=device)

    model.train()
    for _ in range(CFG.linear_epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        losses = ce(logits, y_t)
        loss = (losses * w_t).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def predict_proba_linear_probe(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def fit_oof_and_full(X: np.ndarray, y: np.ndarray, num_classes: int, device: torch.device) -> Tuple[np.ndarray, nn.Module]:
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    oof = np.zeros((len(y), num_classes), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = train_linear_probe(X[tr_idx], y[tr_idx], num_classes, device)
        oof[va_idx] = predict_proba_linear_probe(model, X[va_idx], device)
        print(f"Fold {fold}/{CFG.n_splits} done")

    full_model = train_linear_probe(X, y, num_classes, device)
    return oof, full_model


def tune_model_weights(oof_by_model: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, float]:
    candidates = []
    for d in np.arange(0.30, 0.61, 0.05):
        for e in np.arange(0.20, 0.56, 0.05):
            c = 1.0 - d - e
            if c < 0.10:
                continue
            candidates.append({"dino": round(float(d), 2), "effnet": round(float(e), 2), "clip": round(float(c), 2)})

    best = {
        "dino": CFG.initial_model_weights_dino,
        "effnet": CFG.initial_model_weights_effnet,
        "clip": CFG.initial_model_weights_clip,
    }
    best_acc = -1.0

    for w in candidates:
        p = w["dino"] * oof_by_model["dino"] + w["effnet"] * oof_by_model["effnet"] + w["clip"] * oof_by_model["clip"]
        acc = accuracy_score(y, np.argmax(p, axis=1))
        if acc > best_acc:
            best_acc = acc
            best = w

    print(f"Tuned model weights: {best}, OOF acc={best_acc:.4f}")
    return best


def entropy_of_probs(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def consensus_pseudo_labels(model_probs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preds = {k: np.argmax(v, axis=1) for k, v in model_probs.items()}
    stacked = np.stack([preds["dino"], preds["effnet"], preds["clip"]], axis=1)
    agree = np.all(stacked == stacked[:, :1], axis=1)

    avg_prob = (model_probs["dino"] + model_probs["effnet"] + model_probs["clip"]) / 3.0
    conf = np.max(avg_prob, axis=1)
    ent = entropy_of_probs(avg_prob)

    keep = agree & (conf >= CFG.pseudo_conf_threshold) & (ent <= CFG.pseudo_entropy_threshold)
    idx = np.where(keep)[0]
    labels = np.argmax(avg_prob[idx], axis=1)
    return idx, labels, conf[idx], ent[idx]


def build_meta_learner(X_meta: np.ndarray, y: np.ndarray):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=4000)),
        ]
    )
    grid = GridSearchCV(
        pipe,
        {"clf__C": [0.5, 1.0, 2.0, 5.0], "clf__l1_ratio": [0.2, 0.4, 0.6]},
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_meta, y)
    print(f"Meta best params: {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_


def fit_meta_oof_and_full(X_meta: np.ndarray, y: np.ndarray):
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    classes = np.unique(y)
    num_classes = len(classes)
    oof_prob = np.zeros((len(y), num_classes), dtype=np.float32)
    best_params_by_fold = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_meta, y), start=1):
        model, best_params = build_meta_learner(X_meta[tr_idx], y[tr_idx])
        oof_prob[va_idx] = model.predict_proba(X_meta[va_idx])
        best_params_by_fold.append({"fold": fold, **best_params})
        print(f"Meta OOF fold {fold}/{CFG.n_splits} done")

    final_model, final_best_params = build_meta_learner(X_meta, y)
    return oof_prob, final_model, final_best_params, best_params_by_fold


def compute_classification_metrics(y_true: np.ndarray, prob: np.ndarray, classes: List[str]) -> Dict:
    pred = np.argmax(prob, axis=1)
    cm = confusion_matrix(y_true, pred, labels=list(range(len(classes))))

    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "per_class_recall": {
            cls: float(val)
            for cls, val in zip(classes, recall_score(y_true, pred, average=None, labels=list(range(len(classes))), zero_division=0))
        },
        "per_class_precision": {
            cls: float(val)
            for cls, val in zip(classes, precision_score(y_true, pred, average=None, labels=list(range(len(classes))), zero_division=0))
        },
        "per_class_f1": {
            cls: float(val)
            for cls, val in zip(classes, f1_score(y_true, pred, average=None, labels=list(range(len(classes))), zero_division=0))
        },
        "confusion_matrix": cm.tolist(),
        "true_class_counts": {cls: int((y_true == i).sum()) for i, cls in enumerate(classes)},
        "pred_class_counts": {cls: int((pred == i).sum()) for i, cls in enumerate(classes)},
    }
    return metrics


def save_prediction_csv(path: str, ids: List[str], y_true: np.ndarray, prob: np.ndarray, classes: List[str]) -> None:
    pred = np.argmax(prob, axis=1)
    df = pd.DataFrame({
        "id": ids,
        "y_true": [classes[i] for i in y_true],
        "y_pred": [classes[i] for i in pred],
        "correct": (pred == y_true).astype(int),
    })
    for i, cls in enumerate(classes):
        df[f"prob_{cls}"] = prob[:, i]
    df.to_csv(path, index=False)


def save_confusion_matrix_csv(path: str, y_true: np.ndarray, prob: np.ndarray, classes: List[str]) -> None:
    pred = np.argmax(prob, axis=1)
    cm = confusion_matrix(y_true, pred, labels=list(range(len(classes))))
    df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
    df.to_csv(path)


def summarize_tta(test_probs_tta: Dict[str, List[np.ndarray]], classes: List[str]) -> Dict:
    out = {}
    for name, probs_list in test_probs_tta.items():
        stacked = np.stack(probs_list, axis=0)  # [tta, n, c]
        std_across_tta = np.std(stacked, axis=0)
        mean_std_per_sample = std_across_tta.mean(axis=1)
        out[name] = {
            "mean_probability_std": float(mean_std_per_sample.mean()),
            "median_probability_std": float(np.median(mean_std_per_sample)),
            "max_probability_std": float(mean_std_per_sample.max()),
            "num_tta_views": int(stacked.shape[0]),
            "class_order": classes,
        }
    return out


def config_hash() -> str:
    payload = json.dumps(asdict(CFG), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def maybe_trackio_init(run_name: str):
    if trackio is None:
        return False
    try:
        trackio.init(project="pet-expression-v10", run_name=run_name, config=asdict(CFG))
        return True
    except Exception:
        return False


def maybe_trackio_log(enabled: bool, payload: Dict):
    if not enabled:
        return
    try:
        trackio.log(payload)
    except Exception:
        pass


def maybe_trackio_finish(enabled: bool):
    if not enabled:
        return
    try:
        trackio.finish()
    except Exception:
        pass


def main() -> None:
    set_seed(CFG.seed)
    now_utc = datetime.now(timezone.utc)
    run_id = f"v10-{now_utc.strftime('%Y%m%d-%H%M%S')}"
    run_hash = config_hash()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Running v10 on {device}; run_id={run_id}; cfg={run_hash}")

    trk = maybe_trackio_init(run_id)

    train_dir, test_dir = resolve_data_dirs()
    train_ds = datasets.ImageFolder(train_dir)
    train_paths = [p for p, _ in train_ds.samples]
    train_ids = [os.path.basename(p) for p in train_paths]
    y = np.array([lab for _, lab in train_ds.samples], dtype=np.int64)
    classes = train_ds.classes

    test_files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    test_paths = [os.path.join(test_dir, f) for f in test_files]

    backbones = load_backbones(device)
    tta_keys = ["orig", "flip", "zoom", "zoom_flip"]

    train_features = {}
    test_features = {}

    for name, spec in backbones.items():
        train_features[name] = extract_features(spec, train_paths, "orig", device, "train")
        test_features[name] = {}
        for tta in tta_keys:
            test_features[name][tta] = extract_features(spec, test_paths, tta, device, "test")

    num_classes = len(classes)
    oof_by_model = {}
    full_models = {}

    for name in ["dino", "effnet", "clip"]:
        print(f"Training OOF probes for {name}")
        oof, full_model = fit_oof_and_full(train_features[name], y, num_classes, device)
        oof_by_model[name] = oof
        full_models[name] = full_model

    oof_metrics = {}
    summary_rows = []
    for name in ["dino", "effnet", "clip"]:
        metrics = compute_classification_metrics(y, oof_by_model[name], classes)
        oof_metrics[name] = metrics
        summary_rows.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_recall": metrics["macro_recall"],
            "macro_precision": metrics["macro_precision"],
        })
        print(f"{name}: acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")

    tuned_weights = tune_model_weights(oof_by_model, y)

    weighted_oof_prob = (
        tuned_weights["dino"] * oof_by_model["dino"]
        + tuned_weights["effnet"] * oof_by_model["effnet"]
        + tuned_weights["clip"] * oof_by_model["clip"]
    )
    weighted_oof_metrics = compute_classification_metrics(y, weighted_oof_prob, classes)
    oof_metrics["weighted_ensemble"] = weighted_oof_metrics
    summary_rows.append({
        "model": "weighted_ensemble",
        "accuracy": weighted_oof_metrics["accuracy"],
        "macro_f1": weighted_oof_metrics["macro_f1"],
        "macro_recall": weighted_oof_metrics["macro_recall"],
        "macro_precision": weighted_oof_metrics["macro_precision"],
    })

    oof_stack_input = np.hstack([oof_by_model["dino"], oof_by_model["effnet"], oof_by_model["clip"]])
    meta_oof_prob, meta_model, meta_best_params, meta_best_params_by_fold = fit_meta_oof_and_full(oof_stack_input, y)
    stack_oof_metrics = compute_classification_metrics(y, meta_oof_prob, classes)
    oof_metrics["stack_meta_oof"] = stack_oof_metrics
    summary_rows.append({
        "model": "stack_meta_oof",
        "accuracy": stack_oof_metrics["accuracy"],
        "macro_f1": stack_oof_metrics["macro_f1"],
        "macro_recall": stack_oof_metrics["macro_recall"],
        "macro_precision": stack_oof_metrics["macro_precision"],
    })

    stack_oof_acc = stack_oof_metrics["accuracy"]
    stack_oof_f1 = stack_oof_metrics["macro_f1"]

    maybe_trackio_log(
        trk,
        {
            "stack_oof_acc": stack_oof_acc,
            "stack_oof_macro_f1": stack_oof_f1,
            "dino_oof_acc": oof_metrics["dino"]["accuracy"],
            "effnet_oof_acc": oof_metrics["effnet"]["accuracy"],
            "clip_oof_acc": oof_metrics["clip"]["accuracy"],
            "weighted_oof_acc": weighted_oof_metrics["accuracy"],
        },
    )

    test_probs_tta = {"dino": [], "effnet": [], "clip": []}
    for name in ["dino", "effnet", "clip"]:
        for tta in tta_keys:
            test_probs_tta[name].append(predict_proba_linear_probe(full_models[name], test_features[name][tta], device))

    tta_summary = summarize_tta(test_probs_tta, classes)
    test_probs_by_model = {name: np.mean(np.stack(probs, axis=0), axis=0) for name, probs in test_probs_tta.items()}

    weighted_prob = (
        tuned_weights["dino"] * test_probs_by_model["dino"]
        + tuned_weights["effnet"] * test_probs_by_model["effnet"]
        + tuned_weights["clip"] * test_probs_by_model["clip"]
    )
    weighted_pred = np.argmax(weighted_prob, axis=1)

    stack_test_input = np.hstack([test_probs_by_model["dino"], test_probs_by_model["effnet"], test_probs_by_model["clip"]])
    stack_test_prob = meta_model.predict_proba(stack_test_input)
    stack_pred = np.argmax(stack_test_prob, axis=1)

    pseudo_idx, pseudo_labels, pseudo_conf, pseudo_ent = consensus_pseudo_labels(test_probs_by_model)
    pseudo_count = int(len(pseudo_idx))
    pseudo_ratio = pseudo_count / max(1, len(test_paths))
    pseudo_enabled_this_run = CFG.pseudo_enabled
    pseudo_reason = "enabled"

    if pseudo_enabled_this_run:
        if pseudo_count < CFG.pseudo_min_count:
            pseudo_enabled_this_run = False
            pseudo_reason = f"pseudo_count_low:{pseudo_count}"
        elif pseudo_ratio > CFG.pseudo_max_ratio:
            pseudo_enabled_this_run = False
            pseudo_reason = f"pseudo_ratio_high:{pseudo_ratio:.4f}"

    final_pred = stack_pred.copy()
    final_prob = stack_test_prob.copy()

    if pseudo_enabled_this_run:
        pseudo_models = {}
        for name in ["dino", "effnet", "clip"]:
            X_aug = np.vstack([train_features[name], test_features[name]["orig"][pseudo_idx]])
            y_aug = np.concatenate([y, pseudo_labels])
            sample_w = np.concatenate(
                [np.ones(len(y), dtype=np.float32), np.full(len(pseudo_idx), CFG.pseudo_weight, dtype=np.float32)]
            )
            pseudo_models[name] = train_linear_probe(X_aug, y_aug, num_classes, device, sample_weight=sample_w)

        pseudo_test_probs = {"dino": [], "effnet": [], "clip": []}
        for name in ["dino", "effnet", "clip"]:
            for tta in tta_keys:
                pseudo_test_probs[name].append(predict_proba_linear_probe(pseudo_models[name], test_features[name][tta], device))

        pseudo_test_avg = {name: np.mean(np.stack(probs, axis=0), axis=0) for name, probs in pseudo_test_probs.items()}
        pseudo_stack_input = np.hstack([pseudo_test_avg["dino"], pseudo_test_avg["effnet"], pseudo_test_avg["clip"]])
        final_prob = meta_model.predict_proba(pseudo_stack_input)
        final_pred = np.argmax(final_prob, axis=1)

    # Safety fallback: if OOF stack too low, force weighted baseline
    fallback_triggered = False
    fallback_reason = "none"
    if stack_oof_acc < CFG.min_stack_oof_acc:
        final_pred = weighted_pred
        final_prob = weighted_prob
        fallback_triggered = True
        fallback_reason = f"stack_oof_acc_low:{stack_oof_acc:.4f}"

    # Safety fallback: drift against current team file.
    baseline_path = "submission_final_team.csv"
    drift_ratio = None
    if os.path.exists(baseline_path):
        base_df = pd.read_csv(baseline_path)
        if len(base_df) == len(final_pred):
            base_idx = {k: i for i, k in enumerate(classes)}
            base_pred = np.array([base_idx.get(lbl, -1) for lbl in base_df["label"].tolist()], dtype=np.int64)
            valid = base_pred >= 0
            if valid.mean() > 0.99:
                drift_ratio = float((base_pred != final_pred).mean())
                if drift_ratio > CFG.max_label_shift_ratio:
                    final_pred = stack_pred
                    final_prob = stack_test_prob
                    fallback_triggered = True
                    fallback_reason = f"label_shift_high:{drift_ratio:.4f}"

    # Build outputs
    submission = pd.read_csv("sample_submission.csv")
    submission["id"] = test_files
    submission["label"] = [classes[i] for i in final_pred]
    submission.to_csv("submission_final_team_v10.csv", index=False)

    pseudo_df = pd.DataFrame(
        {
            "id": [test_files[i] for i in pseudo_idx],
            "pseudo_label": [classes[i] for i in pseudo_labels],
            "confidence": pseudo_conf,
            "entropy": pseudo_ent,
            "weight": CFG.pseudo_weight,
        }
    )
    pseudo_df.to_csv("v10_pseudo_labels_consensus.csv", index=False)

    pseudo_distribution = {cls: 0 for cls in classes}
    for lab in pseudo_labels:
        pseudo_distribution[classes[int(lab)]] += 1

    test_detail_df = pd.DataFrame({
        "id": test_files,
        "weighted_pred": [classes[i] for i in weighted_pred],
        "stack_pred": [classes[i] for i in stack_pred],
        "final_pred": [classes[i] for i in final_pred],
    })
    for i, cls in enumerate(classes):
        test_detail_df[f"weighted_prob_{cls}"] = weighted_prob[:, i]
        test_detail_df[f"stack_prob_{cls}"] = stack_test_prob[:, i]
        test_detail_df[f"final_prob_{cls}"] = final_prob[:, i]
    test_detail_df.to_csv("v10_test_predictions_detailed.csv", index=False)

    save_prediction_csv("v10_oof_predictions_dino.csv", train_ids, y, oof_by_model["dino"], classes)
    save_prediction_csv("v10_oof_predictions_effnet.csv", train_ids, y, oof_by_model["effnet"], classes)
    save_prediction_csv("v10_oof_predictions_clip.csv", train_ids, y, oof_by_model["clip"], classes)
    save_prediction_csv("v10_oof_predictions_weighted.csv", train_ids, y, weighted_oof_prob, classes)
    save_prediction_csv("v10_oof_predictions_stack.csv", train_ids, y, meta_oof_prob, classes)

    save_confusion_matrix_csv("v10_confusion_matrix_dino.csv", y, oof_by_model["dino"], classes)
    save_confusion_matrix_csv("v10_confusion_matrix_effnet.csv", y, oof_by_model["effnet"], classes)
    save_confusion_matrix_csv("v10_confusion_matrix_clip.csv", y, oof_by_model["clip"], classes)
    save_confusion_matrix_csv("v10_confusion_matrix_weighted.csv", y, weighted_oof_prob, classes)
    save_confusion_matrix_csv("v10_confusion_matrix_stack.csv", y, meta_oof_prob, classes)

    results_summary_df = pd.DataFrame(summary_rows)
    results_summary_df.to_csv("v10_results_summary.csv", index=False)

    metrics = {
        "run_id": run_id,
        "config_hash": run_hash,
        "train_count": int(len(train_paths)),
        "test_count": int(len(test_paths)),
        "classes": classes,
        "oof": oof_metrics,
        "stack_oof_acc": stack_oof_acc,
        "stack_oof_macro_f1": stack_oof_f1,
        "model_weights": tuned_weights,
        "meta_best_params": meta_best_params,
        "meta_best_params_by_fold": meta_best_params_by_fold,
        "pseudo_count": pseudo_count,
        "pseudo_ratio": pseudo_ratio,
        "pseudo_enabled_config": CFG.pseudo_enabled,
        "pseudo_enabled_this_run": pseudo_enabled_this_run,
        "pseudo_reason": pseudo_reason,
        "pseudo_label_distribution": pseudo_distribution,
        "pseudo_confidence_summary": {
            "mean": float(np.mean(pseudo_conf)) if len(pseudo_conf) > 0 else None,
            "median": float(np.median(pseudo_conf)) if len(pseudo_conf) > 0 else None,
            "min": float(np.min(pseudo_conf)) if len(pseudo_conf) > 0 else None,
            "max": float(np.max(pseudo_conf)) if len(pseudo_conf) > 0 else None,
        },
        "pseudo_entropy_summary": {
            "mean": float(np.mean(pseudo_ent)) if len(pseudo_ent) > 0 else None,
            "median": float(np.median(pseudo_ent)) if len(pseudo_ent) > 0 else None,
            "min": float(np.min(pseudo_ent)) if len(pseudo_ent) > 0 else None,
            "max": float(np.max(pseudo_ent)) if len(pseudo_ent) > 0 else None,
        },
        "tta_summary": tta_summary,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
        "drift_ratio_vs_submission_final_team": drift_ratio,
        "output_files": {
            "final_submission": "submission_final_team_v10.csv",
            "metrics_json": "v10_oof_metrics.json",
            "results_summary_csv": "v10_results_summary.csv",
            "pseudo_labels_csv": "v10_pseudo_labels_consensus.csv",
            "test_predictions_detailed_csv": "v10_test_predictions_detailed.csv",
            "oof_prediction_csvs": [
                "v10_oof_predictions_dino.csv",
                "v10_oof_predictions_effnet.csv",
                "v10_oof_predictions_clip.csv",
                "v10_oof_predictions_weighted.csv",
                "v10_oof_predictions_stack.csv",
            ],
            "confusion_matrix_csvs": [
                "v10_confusion_matrix_dino.csv",
                "v10_confusion_matrix_effnet.csv",
                "v10_confusion_matrix_clip.csv",
                "v10_confusion_matrix_weighted.csv",
                "v10_confusion_matrix_stack.csv",
            ],
        },
    }

    with open("v10_oof_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)

    manifest = {
        "run_id": run_id,
        "timestamp_utc": now_utc.isoformat(),
        "device": str(device),
        "config": asdict(CFG),
        "config_hash": run_hash,
        "inputs": {
            "train_dir": train_dir,
            "test_dir": test_dir,
            "sample_submission": "sample_submission.csv",
        },
        "outputs": {
            "final_submission": "submission_final_team_v10.csv",
            "metrics": "v10_oof_metrics.json",
            "results_summary": "v10_results_summary.csv",
            "pseudo_labels": "v10_pseudo_labels_consensus.csv",
            "test_predictions_detailed": "v10_test_predictions_detailed.csv",
        },
    }

    with open("v10_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    maybe_trackio_log(
        trk,
        {
            "pseudo_count": pseudo_count,
            "pseudo_ratio": pseudo_ratio,
            "fallback_triggered": int(fallback_triggered),
        },
    )
    maybe_trackio_finish(trk)

    print("Generated: submission_final_team_v10.csv")
    print("Generated: v10_oof_metrics.json")
    print("Generated: v10_results_summary.csv")
    print("Generated: v10_test_predictions_detailed.csv")
    print("Generated: v10_run_manifest.json")


if __name__ == "__main__":
    main()
