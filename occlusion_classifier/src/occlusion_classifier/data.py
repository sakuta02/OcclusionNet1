import random
from collections import Counter
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .exceptions import DatasetError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLEAN_CLASS = "Clean"


def image_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def train_transform() -> T.Compose:
    return T.Compose(
        [
            # без vertical flip: кадр с дороги перевернулся бы вверх ногами
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomAutocontrast(p=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def test_transform() -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def build_index(data_root, classes, split, clean_dir=None) -> list[tuple[Path, str]]:
    """Раскладка `<data_root>/<Класс>/<split>/`; папка Clean задаётся отдельным путём."""
    samples: list[tuple[Path, str]] = []
    for class_name in classes:
        paths = image_paths(Path(data_root) / class_name / split)
        if not paths:
            raise DatasetError(f"Нет изображений для класса {class_name} в split={split}")
        samples += [(path, class_name) for path in paths]
    if clean_dir is not None:
        samples += [(path, CLEAN_CLASS) for path in image_paths(clean_dir)]
    return samples


def cap_per_class(samples, max_per_class, seed=42) -> list[tuple[Path, str]]:
    """Датасет несбалансирован — подрезаем каждый класс до одного потолка."""
    rng = random.Random(seed)
    by_class: dict[str, list[Path]] = {}
    for path, class_name in samples:
        by_class.setdefault(class_name, []).append(path)

    capped = []
    for class_name, paths in by_class.items():
        if len(paths) > max_per_class:
            paths = rng.sample(paths, max_per_class)
        capped += [(path, class_name) for path in paths]
    return capped


def class_counts(samples) -> Counter:
    return Counter(class_name for _, class_name in samples)


class OcclusionDataset(Dataset):
    """Multilabel-датасет. Каждый кадр размечен одним классом, поэтому известна
    только одна позиция таргета (positive-unlabeled) — её отмечает `mask`.
    Для `Clean` подтверждены все семь нулей, там маска полная."""

    def __init__(self, samples, classes, transform):
        self.samples = samples
        self.classes = classes
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_name = self.samples[idx]
        image = self.transform(Image.open(path).convert("RGB"))

        target = torch.zeros(len(self.classes))
        mask = torch.zeros(len(self.classes))
        if class_name == CLEAN_CLASS:
            mask[:] = 1.0
        else:
            position = self.class_to_idx[class_name]
            target[position] = 1.0
            mask[position] = 1.0
        return image, target, mask


def build_loaders(config):
    train_samples = build_index(config.data_root, config.classes, "train", config.clean_train_dir)
    test_samples = build_index(config.data_root, config.classes, "test", config.clean_test_dir)
    if config.max_per_class:
        train_samples = cap_per_class(train_samples, config.max_per_class, config.seed)

    train_dataset = OcclusionDataset(train_samples, config.classes, train_transform())
    test_dataset = OcclusionDataset(test_samples, config.classes, test_transform())

    loader_kwargs = {"batch_size": config.batch_size, "num_workers": config.num_workers, "pin_memory": True}
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    )
