import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import test_transform
from .exceptions import CheckpointLoadError
from .models import OcclusionClassifier, build_encoder, load_segmentation_encoder

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BundleManifest:
    schema_version: int = SCHEMA_VERSION
    created_at: str = ""
    classes: list[str] | None = None
    arch: str = "FPN"
    encoder_name: str = "resnet18"
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    threshold: float = 0.5
    metrics: dict[str, Any] | None = None

    def to_dict(self):
        data = asdict(self)
        data["created_at"] = self.created_at or datetime.now(timezone.utc).isoformat()
        return data


def save_bundle(bundle_dir, manifest, state_dict):
    target = Path(bundle_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    torch.save(state_dict, target / "classifier.pt")
    return target


def load_bundle(bundle_dir):
    target = Path(bundle_dir)
    manifest_file = target / "manifest.json"
    if not manifest_file.exists():
        raise CheckpointLoadError(f"Нет manifest.json в {target}")
    manifest = BundleManifest(**json.loads(manifest_file.read_text(encoding="utf-8")))
    state_dict = torch.load(target / "classifier.pt", map_location="cpu", weights_only=True)
    return manifest, state_dict


class OcclusionPredictor:
    """Вероятности по каждому классу и, опционально, карта внимания последнего блока."""

    def __init__(self, model, classes, threshold=0.5, device="cpu"):
        self.model, self.classes, self.threshold, self.device = model, classes, threshold, device
        self.transform = test_transform()

    @classmethod
    def load(cls, bundle_path, *, encoder_checkpoint=None, device="cpu"):
        manifest, state_dict = load_bundle(bundle_path)
        classes = manifest.classes or []
        # веса энкодера уже в бандле, поэтому по умолчанию хватает пустого графа
        encoder = (
            load_segmentation_encoder(manifest.arch, manifest.encoder_name, encoder_checkpoint)
            if encoder_checkpoint
            else build_encoder(manifest.arch, manifest.encoder_name)
        )
        model = OcclusionClassifier(
            encoder,
            num_classes=len(classes),
            embed_dim=manifest.embed_dim,
            num_heads=manifest.num_heads,
            num_layers=manifest.num_layers,
        )
        model.load_state_dict(state_dict)
        return cls(model.to(device).eval(), classes, manifest.threshold, device)

    @torch.inference_mode()
    def predict(self, image_path, return_attn=False):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        if return_attn:
            logits, attention = self.model(tensor, return_attn=True)
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()
            attention = F.interpolate(
                attention, size=image.size[::-1], mode="bilinear", align_corners=False
            )[0].cpu().numpy()
            return dict(zip(self.classes, probabilities.astype(float))), attention
        probabilities = torch.sigmoid(self.model(tensor))[0].cpu().numpy()
        return dict(zip(self.classes, probabilities.astype(float)))

    def detect(self, image_path) -> list[str]:
        probabilities = self.predict(image_path)
        return [name for name, value in probabilities.items() if value > self.threshold]

    def predict_batch(self, image_paths) -> np.ndarray:
        return np.asarray(
            [list(self.predict(path).values()) for path in image_paths], dtype=np.float32
        )
