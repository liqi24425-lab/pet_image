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
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B5_Weights, efficientnet_b0, efficientnet_b5
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
    linear_epochs: int = 40
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
    probe_seeds: Tuple[int, ...] = (42,)
    species_clusters: int = 3


CFG = V10Config()


def apply_env_overrides() -> None:
    # Optional runtime overrides for pseudo-label gating without editing config constants.
    if "V17_PSEUDO_CONF" in os.environ:
        CFG.pseudo_conf_threshold = float(os.environ["V17_PSEUDO_CONF"])
    if "V17_PSEUDO_ENT" in os.environ:
        CFG.pseudo_entropy_threshold = float(os.environ["V17_PSEUDO_ENT"])
    if "V17_PSEUDO_MAX_RATIO" in os.environ:
        CFG.pseudo_max_ratio = float(os.environ["V17_PSEUDO_MAX_RATIO"])
    if "V17_PSEUDO_MIN_COUNT" in os.environ:
        CFG.pseudo_min_count = int(os.environ["V17_PSEUDO_MIN_COUNT"])
    if "V17_PSEUDO_WEIGHT" in os.environ:
        CFG.pseudo_weight = float(os.environ["V17_PSEUDO_WEIGHT"])


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


def load_backbones(device: torch.device, lightweight: bool = False) -> Dict[str, BackboneSpec]:
    tmap = build_transforms()

    if lightweight:
        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        dino_name = "dinov2_vits14"
    else:
        eff = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        dino_name = "dinov2_vitl14"
    eff.classifier[1] = nn.Identity()
    eff = eff.to(device).eval()

    dino = torch.hub.load("facebookresearch/dinov2", dino_name).to(device).eval()
    clip = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32",
        local_files_only=True,
    ).to(device).eval()

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


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    device: torch.device,
    sample_weight: np.ndarray = None,
) -> Pipeline:
    # Keep interface stable while switching to a deterministic, fast CPU classifier.
    del device, num_classes
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", C=1.0, max_iter=800, class_weight="balanced")),
        ]
    )
    pipe.fit(X, y, clf__sample_weight=sample_weight)
    return pipe


def predict_proba_linear_probe(model: Pipeline, X: np.ndarray, device: torch.device) -> np.ndarray:
    del device
    return model.predict_proba(X).astype(np.float32)


def fit_oof_and_full(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    device: torch.device,
    cv_seed: int = 42,
) -> Tuple[np.ndarray, Pipeline]:
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=cv_seed)
    oof = np.zeros((len(y), num_classes), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = train_linear_probe(X[tr_idx], y[tr_idx], num_classes, device)
        oof[va_idx] = predict_proba_linear_probe(model, X[va_idx], device)
        print(f"Fold {fold}/{CFG.n_splits} done")

    full_model = train_linear_probe(X, y, num_classes, device)
    return oof, full_model


def fit_oof_and_full_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    device: torch.device,
    seeds: Tuple[int, ...],
) -> Tuple[np.ndarray, List[Pipeline]]:
    oof_probs = []
    models = []
    for i, sd in enumerate(seeds, start=1):
        print(f"  Seed ensemble member {i}/{len(seeds)} seed={sd}")
        set_seed(sd)
        oof, model = fit_oof_and_full(X, y, num_classes, device, cv_seed=sd)
        oof_probs.append(oof)
        models.append(model)
    oof_mean = np.mean(np.stack(oof_probs, axis=0), axis=0)
    return oof_mean, models


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
    # Use a fixed, fast meta learner to avoid long/unstable saga grid search.
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)),
        ]
    )
    pipe.fit(X_meta, y)
    best_params = {"clf__solver": "lbfgs", "clf__C": 1.0, "clf__max_iter": 1000}
    print(f"Meta params (fixed): {best_params}")
    return pipe, best_params


def build_species_adapters(train_feat: np.ndarray, test_feat: np.ndarray, k: int, seed: int):
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    tr = km.fit_predict(train_feat)
    te = km.predict(test_feat)
    tr_oh = np.eye(k, dtype=np.float32)[tr]
    te_oh = np.eye(k, dtype=np.float32)[te]
    return tr_oh, te_oh


def config_hash() -> str:
    payload = json.dumps(asdict(CFG), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def maybe_trackio_init(run_name: str):
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
    apply_env_overrides()
    set_seed(CFG.seed)
    now_utc = datetime.now(timezone.utc)
    run_id = f"v17-{now_utc.strftime('%Y%m%d-%H%M%S')}"
    run_hash = config_hash()

    feature_device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    train_device = torch.device("cpu")
    print(f"Running v17 on feature={feature_device}, train={train_device}; run_id={run_id}; cfg={run_hash}")

    trk = maybe_trackio_init(run_id)

    train_dir, test_dir = resolve_data_dirs()
    train_ds = datasets.ImageFolder(train_dir)
    train_paths = [p for p, _ in train_ds.samples]
    y = np.array([lab for _, lab in train_ds.samples], dtype=np.int64)
    classes = train_ds.classes

    test_files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    test_paths = [os.path.join(test_dir, f) for f in test_files]

    lightweight = feature_device.type == "cpu"
    full_mode = os.getenv("V17_FULL", "0") == "1"
    if lightweight:
        print("CPU fallback enabled: using lightweight backbones and single TTA.")
    backbones = load_backbones(feature_device, lightweight=lightweight)
    full_pseudo = os.getenv("V17_FULL_PSEUDO", "0") == "1"
    if lightweight:
        tta_keys = ["orig"]
        probe_seeds = (42,)
    elif full_mode:
        tta_keys = ["orig", "flip", "zoom", "zoom_flip"]
        probe_seeds = (42, 52, 62)
        print(f"V17 full mode enabled: tta={tta_keys}, seeds={probe_seeds}, full_pseudo={full_pseudo}")
    else:
        tta_keys = ["orig", "flip"]
        probe_seeds = CFG.probe_seeds

    train_features = {}
    test_features = {}

    for name, spec in backbones.items():
        train_features[name] = extract_features(spec, train_paths, "orig", feature_device, "train")
        test_features[name] = {}
        for tta in tta_keys:
            test_features[name][tta] = extract_features(spec, test_paths, tta, feature_device, "test")

    num_classes = len(classes)
    oof_by_model = {}
    full_models = {}

    for name in ["dino", "effnet", "clip"]:
        print(f"Training OOF probe ensemble for {name}")
        oof, full_model = fit_oof_and_full_ensemble(
            train_features[name],
            y,
            num_classes,
            train_device,
            seeds=probe_seeds,
        )
        oof_by_model[name] = oof
        full_models[name] = full_model

    oof_stats = {}
    for name in ["dino", "effnet", "clip"]:
        pred = np.argmax(oof_by_model[name], axis=1)
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, average="macro"))
        oof_stats[name] = {"acc": acc, "macro_f1": f1}
        print(f"{name}: acc={acc:.4f}, macro_f1={f1:.4f}")

    tuned_weights = tune_model_weights(oof_by_model, y)

    test_probs_tta = {"dino": [], "effnet": [], "clip": []}
    for name in ["dino", "effnet", "clip"]:
        for tta in tta_keys:
            member_probs = [
                predict_proba_linear_probe(model, test_features[name][tta], train_device) for model in full_models[name]
            ]
            test_probs_tta[name].append(np.mean(np.stack(member_probs, axis=0), axis=0))

    test_probs_by_model = {name: np.mean(np.stack(probs, axis=0), axis=0) for name, probs in test_probs_tta.items()}

    weighted_prob = (
        tuned_weights["dino"] * test_probs_by_model["dino"]
        + tuned_weights["effnet"] * test_probs_by_model["effnet"]
        + tuned_weights["clip"] * test_probs_by_model["clip"]
    )
    weighted_pred = np.argmax(weighted_prob, axis=1)
    if lightweight or (full_mode and not full_pseudo):
        # Stable path: use tuned weighted ensemble directly.
        stack_oof_acc = float(accuracy_score(y, np.argmax(
            tuned_weights["dino"] * oof_by_model["dino"]
            + tuned_weights["effnet"] * oof_by_model["effnet"]
            + tuned_weights["clip"] * oof_by_model["clip"],
            axis=1,
        )))
        stack_oof_f1 = float(f1_score(
            y,
            np.argmax(
                tuned_weights["dino"] * oof_by_model["dino"]
                + tuned_weights["effnet"] * oof_by_model["effnet"]
                + tuned_weights["clip"] * oof_by_model["clip"],
                axis=1,
            ),
            average="macro",
        ))
        meta_best_params = {
            "mode": "full_weighted_ensemble" if full_mode and not lightweight else "cpu_fallback_weighted_ensemble"
        }
        pseudo_idx = np.array([], dtype=np.int64)
        pseudo_labels = np.array([], dtype=np.int64)
        pseudo_conf = np.array([], dtype=np.float32)
        pseudo_ent = np.array([], dtype=np.float32)
        pseudo_count = 0
        pseudo_ratio = 0.0
        pseudo_enabled_this_run = False
        pseudo_reason = "disabled_full_weighted_path" if full_mode and not lightweight else "disabled_cpu_fallback"
        stack_pred = weighted_pred.copy()
        final_pred = weighted_pred.copy()
    else:
        if full_mode and full_pseudo:
            print("Stage: species adapters skipped in full+pseudo (stability mode)", flush=True)
            species_train_oh = np.zeros((len(y), 0), dtype=np.float32)
            species_test_oh = np.zeros((len(test_paths), 0), dtype=np.float32)
        else:
            print("Stage: species adapters", flush=True)
            species_train_oh, species_test_oh = build_species_adapters(
                train_features["dino"],
                test_features["dino"]["orig"],
                CFG.species_clusters,
                CFG.seed,
            )

        print("Stage: meta learner fit", flush=True)
        oof_stack_input = np.hstack([oof_by_model["dino"], oof_by_model["effnet"], oof_by_model["clip"], species_train_oh])
        meta_model, meta_best_params = build_meta_learner(oof_stack_input, y)
        stack_oof_pred = meta_model.predict(oof_stack_input)
        stack_oof_acc = float(accuracy_score(y, stack_oof_pred))
        stack_oof_f1 = float(f1_score(y, stack_oof_pred, average="macro"))

        print("Stage: meta predict test", flush=True)
        stack_test_input = np.hstack(
            [test_probs_by_model["dino"], test_probs_by_model["effnet"], test_probs_by_model["clip"], species_test_oh]
        )
        stack_pred = meta_model.predict(stack_test_input)

        print("Stage: pseudo selection", flush=True)
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

        if pseudo_enabled_this_run:
            print(f"Stage: pseudo train start count={len(pseudo_idx)}", flush=True)
            pseudo_models = {}
            for name in ["dino", "effnet", "clip"]:
                X_aug = np.vstack([train_features[name], test_features[name]["orig"][pseudo_idx]])
                y_aug = np.concatenate([y, pseudo_labels])
                sample_w = np.concatenate(
                    [np.ones(len(y), dtype=np.float32), np.full(len(pseudo_idx), CFG.pseudo_weight, dtype=np.float32)]
                )
                print(f"Stage: pseudo train {name}", flush=True)
                pseudo_models[name] = train_linear_probe(X_aug, y_aug, num_classes, train_device, sample_weight=sample_w)

            pseudo_test_probs = {"dino": [], "effnet": [], "clip": []}
            for name in ["dino", "effnet", "clip"]:
                for tta in tta_keys:
                    print(f"Stage: pseudo predict {name}-{tta}", flush=True)
                    pseudo_test_probs[name].append(
                        predict_proba_linear_probe(pseudo_models[name], test_features[name][tta], train_device)
                    )

            pseudo_test_avg = {name: np.mean(np.stack(probs, axis=0), axis=0) for name, probs in pseudo_test_probs.items()}
            pseudo_stack_input = np.hstack(
                [pseudo_test_avg["dino"], pseudo_test_avg["effnet"], pseudo_test_avg["clip"], species_test_oh]
            )
            print("Stage: pseudo meta predict", flush=True)
            final_pred = meta_model.predict(pseudo_stack_input)

    maybe_trackio_log(
        trk,
        {
            "stack_oof_acc": stack_oof_acc,
            "stack_oof_macro_f1": stack_oof_f1,
            "dino_oof_acc": oof_stats["dino"]["acc"],
            "effnet_oof_acc": oof_stats["effnet"]["acc"],
            "clip_oof_acc": oof_stats["clip"]["acc"],
        },
    )

    # Safety fallback: if OOF stack too low, force weighted baseline
    fallback_triggered = False
    fallback_reason = "none"
    if stack_oof_acc < CFG.min_stack_oof_acc:
        final_pred = weighted_pred
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
                    fallback_triggered = True
                    fallback_reason = f"label_shift_high:{drift_ratio:.4f}"

    # Build outputs
    submission = pd.read_csv("sample_submission.csv")
    submission["id"] = test_files
    submission["label"] = [classes[i] for i in final_pred]
    submission.to_csv("submission_final_team_v17.csv", index=False)

    pseudo_df = pd.DataFrame(
        {
            "id": [test_files[i] for i in pseudo_idx],
            "pseudo_label": [classes[i] for i in pseudo_labels],
            "confidence": pseudo_conf,
            "entropy": pseudo_ent,
            "weight": CFG.pseudo_weight,
        }
    )
    pseudo_df.to_csv("v17_pseudo_labels_consensus.csv", index=False)

    metrics = {
        "run_id": run_id,
        "config_hash": run_hash,
        "train_count": int(len(train_paths)),
        "test_count": int(len(test_paths)),
        "classes": classes,
        "oof": oof_stats,
        "stack_oof_acc": stack_oof_acc,
        "stack_oof_macro_f1": stack_oof_f1,
        "model_weights": tuned_weights,
        "meta_best_params": meta_best_params,
        "probe_seeds": list(probe_seeds),
        "full_mode": full_mode,
        "species_clusters": CFG.species_clusters,
        "pseudo_count": pseudo_count,
        "pseudo_ratio": pseudo_ratio,
        "pseudo_enabled_config": CFG.pseudo_enabled,
        "pseudo_enabled_this_run": pseudo_enabled_this_run,
        "pseudo_reason": pseudo_reason,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
        "drift_ratio_vs_submission_final_team": drift_ratio,
    }

    with open("v17_oof_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)

    manifest = {
        "run_id": run_id,
        "timestamp_utc": now_utc.isoformat(),
        "feature_device": str(feature_device),
        "train_device": str(train_device),
        "config": asdict(CFG),
        "config_hash": run_hash,
        "inputs": {
            "train_dir": train_dir,
            "test_dir": test_dir,
            "sample_submission": "sample_submission.csv",
        },
        "outputs": {
            "final_submission": "submission_final_team_v17.csv",
            "metrics": "v17_oof_metrics.json",
            "pseudo_labels": "v17_pseudo_labels_consensus.csv",
        },
    }

    with open("v17_run_manifest.json", "w", encoding="utf-8") as f:
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

    print("Generated: submission_final_team_v17.csv")
    print("Generated: v17_oof_metrics.json")
    print("Generated: v17_run_manifest.json")


if __name__ == "__main__":
    main()
