from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

FEATURE_DIM = 64 + 7 + 128 + 10


def cheap_features(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap_var = cv2.Laplacian(gray, cv2.CV_32F).var() / 10000.0
    edge_density = cv2.Canny(gray.astype(np.uint8), 50, 150).mean() / 255.0
    return np.asarray(
        [
            *image.mean(axis=(0, 1)) / 255.0,
            *image.std(axis=(0, 1)) / 255.0,
            lap_var,
            edge_density,
            gray.mean() / 255.0,
            gray.std() / 255.0,
        ],
        dtype=np.float32,
    )


def build_feature_vector(student, classifier_probs, classifier_semantic, cheap) -> np.ndarray:
    return np.concatenate(
        [np.asarray(part).reshape(-1) for part in (student, classifier_probs, classifier_semantic, cheap)]
    ).astype(np.float32)


def build_feature_matrix(student, classifier_probs, classifier_semantic, cheap) -> np.ndarray:
    return np.concatenate(
        [np.asarray(student), np.asarray(classifier_probs), np.asarray(classifier_semantic), np.asarray(cheap)],
        axis=1,
    ).astype(np.float32)


def extract_classifier_features(model, paths, device="cpu", batch_size=16):
    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    probabilities, semantics = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = torch.stack([transform(Image.open(path).convert("RGB")) for path in batch_paths]).to(device)
            logits, semantic = model.forward_features(images)
            probabilities.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
            semantics.append(semantic.cpu().numpy().astype(np.float32))
    return np.concatenate(probabilities), np.concatenate(semantics)


def extract_dino_features(paths, device="cpu", batch_size=32, components=64):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    teacher = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True).to(device).eval()
    with torch.inference_mode():
        raw = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = torch.stack([transform(Image.open(path).convert("RGB")) for path in batch_paths]).to(device)
            raw.append(teacher(images).cpu().numpy())
    pca = PCA(n_components=components, random_state=42)
    return pca.fit_transform(np.concatenate(raw)).astype(np.float32), pca
