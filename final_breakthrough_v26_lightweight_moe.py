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
from sklearn.metrics import accuracy_score, f1_score, log_loss
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
class V26Config:
    seed: int = 42
    n_splits: int = 5
    batch_size: int = 16
    linear_epochs: int = 40
    linear_lr: float = 3e-3
    linear_weight_decay: float = 1e-3
    label_smoothing: float = 0.08

    pseudo_enabled: bool = False
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
    use_species_adapter: bool = False
    tta_gate_enabled: bool = True
    noise_reweight_enabled: bool = True
    moe_router_enabled: bool = True
    allow_lightweight_stack: bool = True


CFG = V26Config()


def apply_env_overrides() -> None:
    # Optional runtime overrides for pseudo-label gating without editing config constants.
    if "V26_PSEUDO_CONF" in os.environ:
        CFG.pseudo_conf_threshold = float(os.environ["V26_PSEUDO_CONF"])
    elif "V25_PSEUDO_CONF" in os.environ:
        CFG.pseudo_conf_threshold = float(os.environ["V25_PSEUDO_CONF"])
    elif "V22_PSEUDO_CONF" in os.environ:
        CFG.pseudo_conf_threshold = float(os.environ["V22_PSEUDO_CONF"])
    elif "V20_PSEUDO_CONF" in os.environ:
        CFG.pseudo_conf_threshold = float(os.environ["V20_PSEUDO_CONF"])
    elif "V17_PSEUDO_CONF" in os.environ:
        CFG.pseudo_conf_threshold = float(os.environ["V17_PSEUDO_CONF"])
    if "V26_PSEUDO_ENT" in os.environ:
        CFG.pseudo_entropy_threshold = float(os.environ["V26_PSEUDO_ENT"])
    elif "V25_PSEUDO_ENT" in os.environ:
        CFG.pseudo_entropy_threshold = float(os.environ["V25_PSEUDO_ENT"])
    elif "V22_PSEUDO_ENT" in os.environ:
        CFG.pseudo_entropy_threshold = float(os.environ["V22_PSEUDO_ENT"])
    elif "V20_PSEUDO_ENT" in os.environ:
        CFG.pseudo_entropy_threshold = float(os.environ["V20_PSEUDO_ENT"])
    elif "V17_PSEUDO_ENT" in os.environ:
        CFG.pseudo_entropy_threshold = float(os.environ["V17_PSEUDO_ENT"])
    if "V26_PSEUDO_MAX_RATIO" in os.environ:
        CFG.pseudo_max_ratio = float(os.environ["V26_PSEUDO_MAX_RATIO"])
    elif "V25_PSEUDO_MAX_RATIO" in os.environ:
        CFG.pseudo_max_ratio = float(os.environ["V25_PSEUDO_MAX_RATIO"])
    elif "V22_PSEUDO_MAX_RATIO" in os.environ:
        CFG.pseudo_max_ratio = float(os.environ["V22_PSEUDO_MAX_RATIO"])
    elif "V20_PSEUDO_MAX_RATIO" in os.environ:
        CFG.pseudo_max_ratio = float(os.environ["V20_PSEUDO_MAX_RATIO"])
    elif "V17_PSEUDO_MAX_RATIO" in os.environ:
        CFG.pseudo_max_ratio = float(os.environ["V17_PSEUDO_MAX_RATIO"])
    if "V26_PSEUDO_MIN_COUNT" in os.environ:
        CFG.pseudo_min_count = int(os.environ["V26_PSEUDO_MIN_COUNT"])
    elif "V25_PSEUDO_MIN_COUNT" in os.environ:
        CFG.pseudo_min_count = int(os.environ["V25_PSEUDO_MIN_COUNT"])
    elif "V22_PSEUDO_MIN_COUNT" in os.environ:
        CFG.pseudo_min_count = int(os.environ["V22_PSEUDO_MIN_COUNT"])
    elif "V20_PSEUDO_MIN_COUNT" in os.environ:
        CFG.pseudo_min_count = int(os.environ["V20_PSEUDO_MIN_COUNT"])
    elif "V17_PSEUDO_MIN_COUNT" in os.environ:
        CFG.pseudo_min_count = int(os.environ["V17_PSEUDO_MIN_COUNT"])
    if "V26_PSEUDO_WEIGHT" in os.environ:
        CFG.pseudo_weight = float(os.environ["V26_PSEUDO_WEIGHT"])
    elif "V25_PSEUDO_WEIGHT" in os.environ:
        CFG.pseudo_weight = float(os.environ["V25_PSEUDO_WEIGHT"])
    elif "V22_PSEUDO_WEIGHT" in os.environ:
        CFG.pseudo_weight = float(os.environ["V22_PSEUDO_WEIGHT"])
    elif "V20_PSEUDO_WEIGHT" in os.environ:
        CFG.pseudo_weight = float(os.environ["V20_PSEUDO_WEIGHT"])
    elif "V17_PSEUDO_WEIGHT" in os.environ:
        CFG.pseudo_weight = float(os.environ["V17_PSEUDO_WEIGHT"])
    if "V26_TTA_GATE" in os.environ:
        CFG.tta_gate_enabled = os.environ["V26_TTA_GATE"] == "1"
    elif "V25_TTA_GATE" in os.environ:
        CFG.tta_gate_enabled = os.environ["V25_TTA_GATE"] == "1"
    elif "V22_TTA_GATE" in os.environ:
        CFG.tta_gate_enabled = os.environ["V22_TTA_GATE"] == "1"
    if "V26_NOISE_REWEIGHT" in os.environ:
        CFG.noise_reweight_enabled = os.environ["V26_NOISE_REWEIGHT"] == "1"
    elif "V25_NOISE_REWEIGHT" in os.environ:
        CFG.noise_reweight_enabled = os.environ["V25_NOISE_REWEIGHT"] == "1"
    elif "V22_NOISE_REWEIGHT" in os.environ:
        CFG.noise_reweight_enabled = os.environ["V22_NOISE_REWEIGHT"] == "1"
    if "V26_MOE_ROUTER" in os.environ:
        CFG.moe_router_enabled = os.environ["V26_MOE_ROUTER"] == "1"
    elif "V25_MOE_ROUTER" in os.environ:
        CFG.moe_router_enabled = os.environ["V25_MOE_ROUTER"] == "1"
    if "V26_ALLOW_LIGHTWEIGHT_STACK" in os.environ:
        CFG.allow_lightweight_stack = os.environ["V26_ALLOW_LIGHTWEIGHT_STACK"] == "1"


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


class AdaptivePetCrop:
    # ROI path to support off-center heads.
    def __init__(self, resize_to: int = 288, crop_size: int = 224, stride: int = 16):
        self.resize_to = resize_to
        self.crop_size = crop_size
        self.stride = stride

    def __call__(self, image: Image.Image) -> Image.Image:
        img = image.resize((self.resize_to, self.resize_to), Image.Resampling.BILINEAR)
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        sal = gx + gy
        h, w = sal.shape
        win = self.crop_size
        if h <= win or w <= win:
            return img.resize((win, win), Image.Resampling.BILINEAR)

        best_score = -1.0
        best_xy = ((w - win) // 2, (h - win) // 2)
        for y in range(0, h - win + 1, self.stride):
            for x in range(0, w - win + 1, self.stride):
                patch = sal[y : y + win, x : x + win]
                score = float(patch.mean() + 0.2 * patch.std())
                if score > best_score:
                    best_score = score
                    best_xy = (x, y)
        x, y = best_xy
        if best_score < 0.05:
            x, y = (w - win) // 2, (h - win) // 2
        return img.crop((x, y, x + win, y + win))


def build_transforms() -> Dict[str, Dict[str, transforms.Compose]]:
    eff_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    clip_norm = transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711],
    )

    roi = AdaptivePetCrop()

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
            "roi": transforms.Compose([roi, transforms.ToTensor(), norm]),
            "roi_flip": transforms.Compose([roi, transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm]),
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
    sample_weight: np.ndarray = None,
) -> Tuple[np.ndarray, Pipeline]:
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=cv_seed)
    oof = np.zeros((len(y), num_classes), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        tr_w = None if sample_weight is None else sample_weight[tr_idx]
        model = train_linear_probe(X[tr_idx], y[tr_idx], num_classes, device, sample_weight=tr_w)
        oof[va_idx] = predict_proba_linear_probe(model, X[va_idx], device)
        print(f"Fold {fold}/{CFG.n_splits} done")

    full_model = train_linear_probe(X, y, num_classes, device, sample_weight=sample_weight)
    return oof, full_model


def fit_oof_and_full_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    device: torch.device,
    seeds: Tuple[int, ...],
    sample_weight: np.ndarray = None,
) -> Tuple[np.ndarray, List[Pipeline]]:
    oof_probs = []
    models = []
    for i, sd in enumerate(seeds, start=1):
        print(f"  Seed ensemble member {i}/{len(seeds)} seed={sd}")
        set_seed(sd)
        oof, model = fit_oof_and_full(X, y, num_classes, device, cv_seed=sd, sample_weight=sample_weight)
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


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    # Re-normalize probabilities after temperature scaling in logit space.
    p = np.clip(probs, 1e-8, 1.0)
    logits = np.log(p) / max(temperature, 1e-4)
    logits = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(logits)
    return expv / np.clip(expv.sum(axis=1, keepdims=True), 1e-8, None)


def fit_temperature(oof_probs: np.ndarray, y: np.ndarray) -> float:
    best_t = 1.0
    best_nll = float("inf")
    for t in [0.75, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30, 1.50]:
        scaled = apply_temperature(oof_probs, t)
        nll = float(log_loss(y, scaled, labels=np.arange(scaled.shape[1])))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def tune_hybrid_alpha(meta_oof: np.ndarray, weighted_oof: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    best_alpha = 0.0
    best_acc = -1.0
    for alpha in np.arange(0.0, 1.01, 0.05):
        blend = alpha * meta_oof + (1.0 - alpha) * weighted_oof
        acc = float(accuracy_score(y, np.argmax(blend, axis=1)))
        if acc > best_acc:
            best_acc = acc
            best_alpha = float(round(alpha, 2))
    return best_alpha, best_acc


def gate_tta_probs(per_tta_probs: List[np.ndarray], global_keys: List[str], all_keys: List[str]) -> np.ndarray:
    # Multi-view consistency gate inspired by multi-view FER constraints (arXiv:2404.10904).
    stack = np.stack(per_tta_probs, axis=0)  # [T, N, C]
    key_to_idx = {k: i for i, k in enumerate(all_keys)}
    g_idx = [key_to_idx[k] for k in global_keys if k in key_to_idx]
    r_idx = [i for i in range(stack.shape[0]) if i not in g_idx]
    if not g_idx or not r_idx:
        return np.mean(stack, axis=0)

    p_global = np.mean(stack[g_idx], axis=0)
    p_roi = np.mean(stack[r_idx], axis=0)
    pred_g = np.argmax(p_global, axis=1)
    pred_r = np.argmax(p_roi, axis=1)
    conf_g = np.max(p_global, axis=1)
    conf_r = np.max(p_roi, axis=1)

    out = 0.5 * (p_global + p_roi)
    disagree = pred_g != pred_r
    choose_global = disagree & ((conf_g - conf_r) >= 0.08)
    choose_roi = disagree & ((conf_r - conf_g) >= 0.08)
    out[choose_global] = p_global[choose_global]
    out[choose_roi] = p_roi[choose_roi]
    return out


def compute_noise_reweight(oof_by_model: Dict[str, np.ndarray], y: np.ndarray) -> np.ndarray:
    # Noise-aware sample weighting inspired by robust/noisy-label FER (e.g., LA-Net arXiv:2307.09023).
    avg = (oof_by_model["dino"] + oof_by_model["effnet"] + oof_by_model["clip"]) / 3.0
    conf = np.max(avg, axis=1)
    pred = np.argmax(avg, axis=1)
    ent = entropy_of_probs(avg)
    ent_norm = (ent - ent.min()) / (ent.max() - ent.min() + 1e-8)
    conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
    # higher for easy/consistent samples, lower for uncertain or likely noisy labels
    w = 0.6 + 0.7 * conf_norm - 0.4 * ent_norm
    w[pred != y] *= 0.7
    return np.clip(w, 0.35, 1.25).astype(np.float32)


def build_router_features(p_dino: np.ndarray, p_eff: np.ndarray, p_clip: np.ndarray) -> np.ndarray:
    # Low-cost MoE routing features: confidence, entropy and model disagreement.
    pd = np.argmax(p_dino, axis=1)
    pe = np.argmax(p_eff, axis=1)
    pc = np.argmax(p_clip, axis=1)
    agree_all = ((pd == pe) & (pe == pc)).astype(np.float32)
    conf = np.stack([np.max(p_dino, axis=1), np.max(p_eff, axis=1), np.max(p_clip, axis=1)], axis=1)
    ent = np.stack([entropy_of_probs(p_dino), entropy_of_probs(p_eff), entropy_of_probs(p_clip)], axis=1)
    feats = np.hstack([conf, ent, agree_all[:, None], (conf.mean(axis=1) - ent.mean(axis=1))[:, None]])
    return feats.astype(np.float32)


def fit_moe_router(
    y: np.ndarray,
    expert_probs: List[np.ndarray],
    router_feats: np.ndarray,
) -> Tuple[Pipeline, np.ndarray]:
    expert_pred = [np.argmax(p, axis=1) for p in expert_probs]
    expert_conf = [np.max(p, axis=1) for p in expert_probs]
    target = np.zeros(len(y), dtype=np.int64)
    for i in range(len(y)):
        ok = [k for k in range(len(expert_probs)) if expert_pred[k][i] == y[i]]
        if ok:
            target[i] = max(ok, key=lambda k: expert_conf[k][i])
        else:
            target[i] = int(np.argmax([expert_conf[k][i] for k in range(len(expert_probs))]))

    router = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, multi_class="auto")),
        ]
    )
    router.fit(router_feats, target)
    router_prob = router.predict_proba(router_feats).astype(np.float32)
    return router, router_prob


def apply_moe_blend(expert_probs: List[np.ndarray], router_prob: np.ndarray) -> np.ndarray:
    out = np.zeros_like(expert_probs[0], dtype=np.float32)
    for k, p in enumerate(expert_probs):
        out += router_prob[:, k : k + 1] * p
    return out


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
    run_id = f"v26-{now_utc.strftime('%Y%m%d-%H%M%S')}"
    run_hash = config_hash()

    feature_device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    train_device = torch.device("cpu")
    print(f"Running v26 on feature={feature_device}, train={train_device}; run_id={run_id}; cfg={run_hash}")

    trk = maybe_trackio_init(run_id)

    train_dir, test_dir = resolve_data_dirs()
    train_ds = datasets.ImageFolder(train_dir)
    train_paths = [p for p, _ in train_ds.samples]
    y = np.array([lab for _, lab in train_ds.samples], dtype=np.int64)
    classes = train_ds.classes

    test_files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    test_paths = [os.path.join(test_dir, f) for f in test_files]

    lightweight = feature_device.type == "cpu"
    full_mode = os.getenv(
        "V26_FULL", os.getenv("V25_FULL", os.getenv("V22_FULL", os.getenv("V20_FULL", os.getenv("V17_FULL", "0"))))
    ) == "1"
    if lightweight:
        print("CPU fallback enabled: using lightweight backbones and single TTA.")
    backbones = load_backbones(feature_device, lightweight=lightweight)
    full_pseudo = os.getenv(
        "V26_FULL_PSEUDO",
        os.getenv(
            "V25_FULL_PSEUDO", os.getenv("V22_FULL_PSEUDO", os.getenv("V20_FULL_PSEUDO", os.getenv("V17_FULL_PSEUDO", "0")))
        ),
    ) == "1"
    cpu_fallback_mode = lightweight and not CFG.allow_lightweight_stack
    if cpu_fallback_mode:
        tta_keys = ["orig"]
        probe_seeds = (42,)
    elif full_mode:
        tta_keys = ["orig", "flip", "zoom", "zoom_flip", "roi", "roi_flip"]
        probe_seeds = (42, 52, 62)
        print(f"V26 full mode enabled: tta={tta_keys}, seeds={probe_seeds}, full_pseudo={full_pseudo}")
    elif lightweight and CFG.allow_lightweight_stack:
        # CPU-usable richer path: keep lightweight backbones but enable stacking and router.
        tta_keys = ["orig", "flip", "roi"]
        probe_seeds = (42, 52)
        print(f"V26 lightweight-stack mode: tta={tta_keys}, seeds={probe_seeds}, full_pseudo={full_pseudo}")
    else:
        tta_keys = ["orig", "flip", "roi"]
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

    train_sample_weight = np.ones(len(y), dtype=np.float32)
    if CFG.noise_reweight_enabled:
        train_sample_weight = compute_noise_reweight(oof_by_model, y)
        print(
            f"Noise reweight enabled: min={train_sample_weight.min():.3f}, "
            f"mean={train_sample_weight.mean():.3f}, max={train_sample_weight.max():.3f}"
        )
        # Refit probes with noise-aware sample weights (single extra pass).
        for name in ["dino", "effnet", "clip"]:
            print(f"Refit noise-aware OOF probe ensemble for {name}")
            oof, full_model = fit_oof_and_full_ensemble(
                train_features[name],
                y,
                num_classes,
                train_device,
                seeds=probe_seeds,
                sample_weight=train_sample_weight,
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

    # Per-backbone temperature calibration on OOF probabilities.
    model_temps = {name: fit_temperature(oof_by_model[name], y) for name in ["dino", "effnet", "clip"]}
    calibrated_oof = {name: apply_temperature(oof_by_model[name], model_temps[name]) for name in ["dino", "effnet", "clip"]}
    print(f"Calibrated temperatures: {model_temps}")

    tuned_weights = tune_model_weights(calibrated_oof, y)

    test_probs_tta = {"dino": [], "effnet": [], "clip": []}
    for name in ["dino", "effnet", "clip"]:
        for tta in tta_keys:
            member_probs = [
                predict_proba_linear_probe(model, test_features[name][tta], train_device) for model in full_models[name]
            ]
            test_probs_tta[name].append(np.mean(np.stack(member_probs, axis=0), axis=0))

    global_tta_keys = [k for k in ["orig", "flip", "zoom", "zoom_flip"] if k in tta_keys]
    if CFG.tta_gate_enabled:
        test_probs_by_model = {
            name: apply_temperature(gate_tta_probs(probs, global_tta_keys, tta_keys), model_temps[name])
            for name, probs in test_probs_tta.items()
        }
    else:
        test_probs_by_model = {
            name: apply_temperature(np.mean(np.stack(probs, axis=0), axis=0), model_temps[name])
            for name, probs in test_probs_tta.items()
        }

    weighted_prob = (
        tuned_weights["dino"] * test_probs_by_model["dino"]
        + tuned_weights["effnet"] * test_probs_by_model["effnet"]
        + tuned_weights["clip"] * test_probs_by_model["clip"]
    )
    weighted_pred = np.argmax(weighted_prob, axis=1)
    router_used = False
    router_oof_acc = None
    if cpu_fallback_mode:
        # Stable path: use tuned weighted ensemble directly on CPU fallback.
        stack_oof_acc = float(accuracy_score(y, np.argmax(
            tuned_weights["dino"] * calibrated_oof["dino"]
            + tuned_weights["effnet"] * calibrated_oof["effnet"]
            + tuned_weights["clip"] * calibrated_oof["clip"],
            axis=1,
        )))
        stack_oof_f1 = float(f1_score(
            y,
            np.argmax(
                tuned_weights["dino"] * calibrated_oof["dino"]
                + tuned_weights["effnet"] * calibrated_oof["effnet"]
                + tuned_weights["clip"] * calibrated_oof["clip"],
                axis=1,
            ),
            average="macro",
        ))
        meta_best_params = {"mode": "cpu_fallback_weighted_ensemble"}
        hybrid_alpha = 0.0
        pseudo_idx = np.array([], dtype=np.int64)
        pseudo_labels = np.array([], dtype=np.int64)
        pseudo_conf = np.array([], dtype=np.float32)
        pseudo_ent = np.array([], dtype=np.float32)
        pseudo_count = 0
        pseudo_ratio = 0.0
        pseudo_enabled_this_run = False
        pseudo_reason = "disabled_cpu_fallback"
        stack_pred = weighted_pred.copy()
        final_pred = weighted_pred.copy()
    else:
        if not CFG.use_species_adapter:
            print("Stage: species adapters disabled (v26 default)", flush=True)
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
        oof_stack_input = np.hstack(
            [calibrated_oof["dino"], calibrated_oof["effnet"], calibrated_oof["clip"], species_train_oh]
        )
        meta_model, meta_best_params = build_meta_learner(oof_stack_input, y)
        meta_oof_prob = meta_model.predict_proba(oof_stack_input).astype(np.float32)
        weighted_oof_prob = (
            tuned_weights["dino"] * calibrated_oof["dino"]
            + tuned_weights["effnet"] * calibrated_oof["effnet"]
            + tuned_weights["clip"] * calibrated_oof["clip"]
        )
        consistency_oof_prob = (calibrated_oof["dino"] + calibrated_oof["effnet"] + calibrated_oof["clip"]) / 3.0
        hybrid_alpha, stack_oof_acc = tune_hybrid_alpha(meta_oof_prob, weighted_oof_prob, y)
        blend_oof = hybrid_alpha * meta_oof_prob + (1.0 - hybrid_alpha) * weighted_oof_prob
        stack_oof_f1 = float(f1_score(y, np.argmax(blend_oof, axis=1), average="macro"))
        print(f"Hybrid alpha={hybrid_alpha:.2f}, OOF acc={stack_oof_acc:.4f}, OOF macro_f1={stack_oof_f1:.4f}")

        router_used = False
        router_oof_acc = None

        print("Stage: meta predict test", flush=True)
        stack_test_input = np.hstack(
            [test_probs_by_model["dino"], test_probs_by_model["effnet"], test_probs_by_model["clip"], species_test_oh]
        )
        meta_test_prob = meta_model.predict_proba(stack_test_input).astype(np.float32)
        weighted_test_prob = weighted_prob
        consistency_test_prob = (test_probs_by_model["dino"] + test_probs_by_model["effnet"] + test_probs_by_model["clip"]) / 3.0
        blend_test_prob = hybrid_alpha * meta_test_prob + (1.0 - hybrid_alpha) * weighted_test_prob

        if CFG.moe_router_enabled:
            print("Stage: moe router fit", flush=True)
            router_feats_oof = build_router_features(calibrated_oof["dino"], calibrated_oof["effnet"], calibrated_oof["clip"])
            router_model, router_oof_prob = fit_moe_router(
                y,
                expert_probs=[weighted_oof_prob, meta_oof_prob, consistency_oof_prob],
                router_feats=router_feats_oof,
            )
            blend_oof_router = apply_moe_blend([weighted_oof_prob, meta_oof_prob, consistency_oof_prob], router_oof_prob)
            router_oof_acc = float(accuracy_score(y, np.argmax(blend_oof_router, axis=1)))
            if router_oof_acc >= stack_oof_acc:
                stack_oof_acc = router_oof_acc
                stack_oof_f1 = float(f1_score(y, np.argmax(blend_oof_router, axis=1), average="macro"))
                print(f"MOE router accepted: OOF acc={stack_oof_acc:.4f}, macro_f1={stack_oof_f1:.4f}")
                router_feats_test = build_router_features(
                    test_probs_by_model["dino"], test_probs_by_model["effnet"], test_probs_by_model["clip"]
                )
                router_test_prob = router_model.predict_proba(router_feats_test).astype(np.float32)
                blend_test_prob = apply_moe_blend([weighted_test_prob, meta_test_prob, consistency_test_prob], router_test_prob)
                router_used = True
            else:
                print(f"MOE router rejected: router_oof={router_oof_acc:.4f} < hybrid_oof={stack_oof_acc:.4f}")
        stack_pred = np.argmax(blend_test_prob, axis=1)

        print("Stage: pseudo selection", flush=True)
        pseudo_idx, pseudo_labels, pseudo_conf, pseudo_ent = consensus_pseudo_labels(test_probs_by_model)
        pseudo_count = int(len(pseudo_idx))
        pseudo_ratio = pseudo_count / max(1, len(test_paths))
        pseudo_enabled_this_run = CFG.pseudo_enabled
        pseudo_reason = "enabled" if pseudo_enabled_this_run else "disabled_by_config"

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

            if CFG.tta_gate_enabled:
                pseudo_test_avg = {
                    name: apply_temperature(gate_tta_probs(probs, global_tta_keys, tta_keys), model_temps[name])
                    for name, probs in pseudo_test_probs.items()
                }
            else:
                pseudo_test_avg = {
                    name: apply_temperature(np.mean(np.stack(probs, axis=0), axis=0), model_temps[name])
                    for name, probs in pseudo_test_probs.items()
                }
            pseudo_stack_input = np.hstack(
                [pseudo_test_avg["dino"], pseudo_test_avg["effnet"], pseudo_test_avg["clip"], species_test_oh]
            )
            print("Stage: pseudo meta predict", flush=True)
            pseudo_meta_prob = meta_model.predict_proba(pseudo_stack_input).astype(np.float32)
            pseudo_weighted_prob = (
                tuned_weights["dino"] * pseudo_test_avg["dino"]
                + tuned_weights["effnet"] * pseudo_test_avg["effnet"]
                + tuned_weights["clip"] * pseudo_test_avg["clip"]
            )
            final_pred = np.argmax(hybrid_alpha * pseudo_meta_prob + (1.0 - hybrid_alpha) * pseudo_weighted_prob, axis=1)

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
    submission.to_csv("submission_final_team_v26.csv", index=False)

    pseudo_df = pd.DataFrame(
        {
            "id": [test_files[i] for i in pseudo_idx],
            "pseudo_label": [classes[i] for i in pseudo_labels],
            "confidence": pseudo_conf,
            "entropy": pseudo_ent,
            "weight": CFG.pseudo_weight,
        }
    )
    pseudo_df.to_csv("v26_pseudo_labels_consensus.csv", index=False)

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
        "model_temperatures": model_temps,
        "hybrid_alpha": hybrid_alpha,
        "moe_router_enabled": CFG.moe_router_enabled,
        "moe_router_used": router_used,
        "moe_router_oof_acc": router_oof_acc,
        "meta_best_params": meta_best_params,
        "probe_seeds": list(probe_seeds),
        "full_mode": full_mode,
        "full_pseudo_mode": full_pseudo,
        "tta_gate_enabled": CFG.tta_gate_enabled,
        "noise_reweight_enabled": CFG.noise_reweight_enabled,
        "train_sample_weight_stats": {
            "min": float(train_sample_weight.min()),
            "mean": float(train_sample_weight.mean()),
            "max": float(train_sample_weight.max()),
        },
        "species_clusters": CFG.species_clusters,
        "species_adapter_enabled": CFG.use_species_adapter,
        "pseudo_count": pseudo_count,
        "pseudo_ratio": pseudo_ratio,
        "pseudo_enabled_config": CFG.pseudo_enabled,
        "pseudo_enabled_this_run": pseudo_enabled_this_run,
        "pseudo_reason": pseudo_reason,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
        "drift_ratio_vs_submission_final_team": drift_ratio,
    }

    with open("v26_oof_metrics.json", "w", encoding="utf-8") as f:
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
            "final_submission": "submission_final_team_v26.csv",
            "metrics": "v26_oof_metrics.json",
            "pseudo_labels": "v26_pseudo_labels_consensus.csv",
        },
    }

    with open("v26_run_manifest.json", "w", encoding="utf-8") as f:
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

    print("Generated: submission_final_team_v26.csv")
    print("Generated: v26_oof_metrics.json")
    print("Generated: v26_run_manifest.json")


if __name__ == "__main__":
    main()
