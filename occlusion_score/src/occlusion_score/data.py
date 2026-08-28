import re
from pathlib import Path
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def group_key(path: str | Path) -> str:
    return re.sub(r"([_-]frame[_-]?\d+).*?$", "", Path(path).stem, flags=re.IGNORECASE)


def recover_labels(labels_root: str | Path, dataset_root: str | Path):
    dataset = {path.name: path for path in image_paths(dataset_root)}
    rows = []
    for label_file in Path(labels_root).rglob("*.csv"):
        frame = pd.read_csv(label_file)
        for filename, score in zip(frame["filename"], frame["occlusion_score"]):
            rows.append((str(dataset[filename]), float(score)))

    labels = pd.DataFrame(rows, columns=["image_path", "score"])
    result = labels.groupby("image_path", as_index=False).agg(
        human_score=("score", "mean"),
        human_std=("score", "std"),
        n_raters=("score", "count"),
    )
    result["human_std"] = result["human_std"].fillna(0.0)
    result["weight"] = 1.0 / (1.0 + 5.0 * result["human_std"])
    result["group"] = result["image_path"].map(group_key)
    return result
