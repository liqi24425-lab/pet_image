import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B5_Weights, efficientnet_b5
from transformers import CLIPVisionModelWithProjection
from tqdm import tqdm

BATCH_SIZE = 16
EPOCHS = 120
LR = 3e-3
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.08
PSEUDO_CONF_THRESHOLD = 0.90
PSEUDO_ENTROPY_THRESHOLD = 0.25
PSEUDO_WEIGHT = 0.3


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


def resolve_data_dirs() -> Tuple[str, str]:
    if os.path.isdir("train/train"):
        train_dir = "train/train"
    elif os.path.isdir("train"):
        train_dir = "train"
    else:
        raise FileNotFoundError("Cannot find train directory.")

    if os.path.isdir("test/test"):
        test_dir = "test/test"
    elif os.path.isdir("test"):
        test_dir = "test"
    else:
        raise FileNotFoundError("Cannot find test directory.")

    return train_dir, test_dir


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

    return {
        "effnet": make_ttas(eff_norm),
        "dino": make_ttas(eff_norm),
        "clip": make_ttas(clip_norm),
    }


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


def extract_features(spec: BackboneSpec, paths: List[str], tta_key: str, device: torch.device) -> np.ndarray:
    loader = DataLoader(PathDataset(paths, spec.transform_map[tta_key]), batch_size=BATCH_SIZE, shuffle=False)
    feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{spec.name}-{tta_key}"):
            batch = batch.to(device)
            if spec.name == "clip":
                out = spec.model(pixel_values=batch).image_embeds
            else:
                out = spec.model(batch)
            feats.append(out.cpu().numpy())
    return np.vstack(feats)


def class_weights(y: np.ndarray, n_class: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_class).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)


def train_linear_probe(X: np.ndarray, y: np.ndarray, n_class: int, device: torch.device) -> nn.Module:
    probe = nn.Linear(X.shape[1], n_class).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss(weight=class_weights(y, n_class).to(device), label_smoothing=LABEL_SMOOTHING)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    probe.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = probe(X_t)
        loss = ce(logits, y_t)
        loss.backward()
        opt.step()

    probe.eval()
    return probe


def predict_proba(probe: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        logits = probe(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def entropy_of_probs(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Pseudo-label stage on device: {device}")

    train_dir, test_dir = resolve_data_dirs()
    train_ds = datasets.ImageFolder(train_dir)
    classes = train_ds.classes
    y = np.array([lab for _, lab in train_ds.samples], dtype=np.int64)
    train_paths = [p for p, _ in train_ds.samples]

    test_files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    test_paths = [os.path.join(test_dir, f) for f in test_files]

    backbones = load_backbones(device)
    tta_keys = ["orig", "flip", "zoom", "zoom_flip"]

    train_features = {}
    test_features = {}
    for name, spec in backbones.items():
        train_features[name] = extract_features(spec, train_paths, "orig", device)
        test_features[name] = {tta: extract_features(spec, test_paths, tta, device) for tta in tta_keys}

    model_probs = {}
    for name in ["dino", "effnet", "clip"]:
        probe = train_linear_probe(train_features[name], y, len(classes), device)
        probs_tta = [predict_proba(probe, test_features[name][tta], device) for tta in tta_keys]
        model_probs[name] = np.mean(np.stack(probs_tta, axis=0), axis=0)

    preds = {k: np.argmax(v, axis=1) for k, v in model_probs.items()}
    votes = np.stack([preds["dino"], preds["effnet"], preds["clip"]], axis=1)
    agree = np.all(votes == votes[:, :1], axis=1)

    mean_prob = (model_probs["dino"] + model_probs["effnet"] + model_probs["clip"]) / 3.0
    conf = np.max(mean_prob, axis=1)
    ent = entropy_of_probs(mean_prob)

    keep = agree & (conf >= PSEUDO_CONF_THRESHOLD) & (ent <= PSEUDO_ENTROPY_THRESHOLD)
    keep_idx = np.where(keep)[0]
    keep_labels = np.argmax(mean_prob[keep_idx], axis=1)

    out = pd.DataFrame(
        {
            "id": [test_files[i] for i in keep_idx],
            "pseudo_label": [classes[i] for i in keep_labels],
            "confidence": conf[keep_idx],
            "entropy": ent[keep_idx],
            "weight": PSEUDO_WEIGHT,
        }
    )
    out.to_csv("pseudo_labels_consensus.csv", index=False)
    print(f"Saved pseudo_labels_consensus.csv with {len(out)} rows")


if __name__ == "__main__":
    main()
