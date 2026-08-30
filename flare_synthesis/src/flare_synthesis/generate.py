import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from .dataset import FlareCompositeDataset


@dataclass
class GenerationResult:
    output_dir: Path
    n_generated: int
    archive_path: Path | None = None


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def center_crop(image, size):
    width, height = image.size
    left, top = (width - size) // 2, (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def run_generation(config) -> GenerationResult:
    device = resolve_device(config.device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # размер кадра фиксирован, это ускоряет conv/blur
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataset = FlareCompositeDataset(
        config.background_dir, config.flare_dir, config.light_dir, config.img_size, device
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(config.n_samples):
        # сохраняем только lq: gt и flare нужны для restoration, классификатору — нет
        sample = dataset[random.randrange(len(dataset))]
        target = output_dir / f"{i:05d}.png"
        save_image(sample["lq"], target)
        if config.crop_size:
            center_crop(Image.open(target), config.crop_size).save(target)

    archive_path = None
    if config.archive:
        archive_path = Path(shutil.make_archive(str(output_dir), "zip", output_dir))
    return GenerationResult(output_dir, config.n_samples, archive_path)
