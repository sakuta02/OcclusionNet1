import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from occlusion_score.data import image_paths
from occlusion_score.inference import OcclusionScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("images", type=Path)
    args = parser.parse_args()
    scorer = OcclusionScorer.load(args.bundle)
    paths = image_paths(args.images) if args.images.is_dir() else [args.images]
    writer = csv.writer(sys.stdout)
    writer.writerow(["image_path", "occlusion_score"])
    for path, score in zip(paths, scorer.predict_batch(paths)):
        writer.writerow([path, f"{score:.6f}"])


if __name__ == "__main__":
    main()
