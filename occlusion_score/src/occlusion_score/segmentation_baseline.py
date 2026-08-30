"""Ранний вариант скора загрязнённости: сегментационная модель размечает кадр
на четыре порядковых класса (Clear / Transparent / Semi / Opaque), скор — их
взвешенное среднее по пикселям. Живёт рядом с основным (diff-based) скором как
бейзлайн для сравнения."""

import re
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image

from .exceptions import OcclusionScoreError

ARCHITECTURES = {
    "fpn": smp.FPN,
    "linknet": smp.Linknet,
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "pan": smp.PAN,
    "pspnet": smp.PSPNet,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "deeplabv3": smp.DeepLabV3,
}

# Clear / Transparent / Semi / Opaque — порядковая шкала, отсюда равномерные веса
ORDINAL_WEIGHTS = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])

PALETTE = {0: (34, 139, 34), 1: (255, 215, 0), 2: (255, 140, 0), 3: (220, 20, 60)}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def ordinal_occlusion_score(logits) -> float:
    """Средняя по кадру ожидаемая степень загрязнённости из softmax по классам."""
    probabilities = torch.softmax(logits, dim=1)
    weights = ORDINAL_WEIGHTS.to(probabilities.device).view(1, -1, 1, 1)
    return (probabilities * weights).sum(dim=1).mean().item()


def parse_model_name(folder_name) -> tuple[str, str]:
    """`fpn_resnet18_focal_loss_all_files` -> ('fpn', 'resnet18')."""
    for arch in sorted(ARCHITECTURES, key=len, reverse=True):
        if folder_name.startswith(f"{arch}_"):
            match = re.match(r"(resnet\d+)", folder_name[len(arch) + 1 :])
            return arch, match.group(1) if match else "resnet18"
    raise OcclusionScoreError(f"Не смог распознать архитектуру в {folder_name}")


def load_baseline(model_root, folder_name, n_classes=4):
    arch, encoder = parse_model_name(folder_name)
    model = ARCHITECTURES[arch](encoder_name=encoder, encoder_weights=None, classes=n_classes)

    checkpoint_path = Path(model_root) / folder_name / "model" / f"{arch}_{encoder}" / f"{arch}_{encoder}.ckpt"
    # weights_only=False — чекпойнт свой, не из внешнего источника
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("state_dict", checkpoint)
    state_dict = {(key[len("model.") :] if key.startswith("model.") else key): value for key, value in raw.items()}

    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def preprocess(image_path, size=(480, 640)):
    image = Image.open(image_path).convert("RGB").resize(size[::-1])
    array = (np.array(image).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return image, torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()


def mask_to_rgb(mask) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_index, color in PALETTE.items():
        rgb[mask == class_index] = color
    return rgb


@torch.inference_mode()
def score_image(model, image_path, size=(480, 640)) -> tuple[float, np.ndarray]:
    _, tensor = preprocess(image_path, size)
    logits = model(tensor)
    return ordinal_occlusion_score(logits), torch.argmax(logits, dim=1).squeeze(0).numpy()
