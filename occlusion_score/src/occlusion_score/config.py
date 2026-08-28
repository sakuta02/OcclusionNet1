from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    weights_path: Path
    labels_root: Path
    output_dir: Path
    seed: int = 42
    device: str = "auto"
    n_seeds: int = 5
    n_splits: int = 5

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TrainingConfig":
        converted = {
            key: Path(value) if key.endswith("_root") or key.endswith("_path") or key == "output_dir" else value
            for key, value in values.items()
        }
        return cls(**converted)
