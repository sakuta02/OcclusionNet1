from dataclasses import dataclass
from pathlib import Path
from typing import Any

PATH_KEYS = {"background_dir", "flare_dir", "light_dir", "output_dir"}


@dataclass(frozen=True)
class GenerationConfig:
    background_dir: Path
    flare_dir: Path
    light_dir: Path
    output_dir: Path
    n_samples: int = 1000
    img_size: int = 720  # высота кадра BDD100K — синтез идёт без паддинга
    crop_size: int | None = 512
    archive: bool = False
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "GenerationConfig":
        converted = {
            key: (Path(value) if key in PATH_KEYS and value is not None else value)
            for key, value in values.items()
        }
        return cls(**converted)
