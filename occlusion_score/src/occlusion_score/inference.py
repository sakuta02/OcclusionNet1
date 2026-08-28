import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import joblib
import numpy as np
import torch
import torchvision.models as tvm
from PIL import Image
from torchvision import transforms

from .exceptions import BundleLoadError
from .features import FEATURE_DIM, build_feature_vector, cheap_features
from .models import ExactClassifier, MobileNetStudent, TinyHead

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BundleManifest:
    schema_version: int = SCHEMA_VERSION
    status: str = "REJECT"
    created_at: str = ""
    feature_dim: int = FEATURE_DIM
    metrics: dict[str, Any] | None = None
    classifier_classes: list[str] | None = None

    def to_dict(self):
        data = asdict(self)
        data["created_at"] = self.created_at or datetime.now(timezone.utc).isoformat()
        return data


def save_bundle(bundle_dir, manifest, artifacts, torch_artifacts=None):
    target = Path(bundle_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    joblib.dump(artifacts, target / "sklearn.joblib")
    if torch_artifacts is not None:
        torch.save(torch_artifacts, target / "torch.pt")
    return target


def load_bundle(bundle_dir):
    target = Path(bundle_dir)
    manifest_data = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest_data["status"] != "ACCEPT":
        raise BundleLoadError("Bundle is not production-approved")
    artifacts = joblib.load(target / "sklearn.joblib")
    artifacts["torch"] = torch.load(target / "torch.pt", map_location="cpu", weights_only=True)
    return BundleManifest(**manifest_data), artifacts


class OcclusionScorer:
    def __init__(self, classifier_features: Callable, student_features: Callable, heads, isotonic):
        self.classifier_features, self.student_features = classifier_features, student_features
        self.heads, self.isotonic = heads, isotonic

    @classmethod
    def load(cls, bundle_path, *, classifier_features=None, student_features=None, device="cpu"):
        manifest, artifacts = load_bundle(bundle_path)

        if classifier_features is None or student_features is None:
            classifier = ExactClassifier(tvm.efficientnet_b3(weights=None).features, len(manifest.classifier_classes or []))
            classifier.load_state_dict(artifacts["torch"]["classifier_state"])
            classifier.to(device).eval()

            student = MobileNetStudent(weights=None)
            student.load_state_dict(artifacts["torch"]["student_state"])
            student.to(device).eval()

            clf_transform = transforms.Compose(
                [
                    transforms.Resize((300, 300)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            student_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            def classifier_features(path):
                with torch.inference_mode():
                    image = clf_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                    logits, semantic = classifier.forward_features(image)
                return torch.sigmoid(logits).cpu().numpy().ravel(), semantic.cpu().numpy().ravel()

            def student_features(path):
                with torch.inference_mode():
                    image = student_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                    return student(image).cpu().numpy().ravel()

        heads = []
        for scaler, state in zip(artifacts["scalers"], artifacts["torch"]["head_states"]):
            head = TinyHead()
            head.load_state_dict(state)
            head.eval()
            heads.append((scaler, head))

        return cls(classifier_features, student_features, heads, artifacts["isotonic"])

    def predict(self, image_path):
        probs, semantic = self.classifier_features(image_path)
        vector = build_feature_vector(self.student_features(image_path), probs, semantic, cheap_features(image_path))
        values = [
            head(torch.from_numpy(scaler.transform(vector[None, :]).astype(np.float32))).item()
            for scaler, head in self.heads
        ]
        return float(np.clip(self.isotonic.predict([float(np.mean(values))])[0], 0.0, 1.0))

    def predict_batch(self, image_paths):
        return np.asarray([self.predict(path) for path in image_paths], dtype=np.float32)
