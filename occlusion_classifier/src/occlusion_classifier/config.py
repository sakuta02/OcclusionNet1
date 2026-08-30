from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CLASSES = [
    "DaytimeFlare",
    "Fog",
    "MotionBlur",
    "NighttimeFlare",
    "Raindrops",
    "Soil",
]

PATH_KEYS = {"data_root", "clean_train_dir", "clean_test_dir", "encoder_checkpoint", "output_dir", "inference_dir"}


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    encoder_checkpoint: Path
    output_dir: Path
    clean_train_dir: Path | None = None
    clean_test_dir: Path | None = None
    inference_dir: Path | None = None
    classes: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    arch: str = "FPN"
    encoder_name: str = "resnet18"
    freeze_encoder: bool = False
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.2
    epochs: int = 30
    batch_size: int = 32
    num_workers: int = 4
    encoder_lr: float = 1e-5
    head_lr: float = 1e-4
    focal_gamma: float = 2.0
    max_per_class: int | None = 800
    threshold: float = 0.5
    seed: int = 42
    device: str = "auto"
    project_name: str = "OcclusionNet"
    task_name: str = "FPN_resnet18_multilabel"
    inference_every_n_epochs: int = 1

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TrainingConfig":
        converted = {
            key: (Path(value) if key in PATH_KEYS and value is not None else value)
            for key, value in values.items()
        }
        return cls(**converted)
