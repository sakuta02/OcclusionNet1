import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from occlusion_classifier.data import image_paths
from occlusion_classifier.inference import OcclusionPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("images", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    predictor = OcclusionPredictor.load(args.bundle, device=args.device)
    paths = image_paths(args.images) if args.images.is_dir() else [args.images]

    writer = csv.writer(sys.stdout)
    writer.writerow(["image_path", *predictor.classes, "detected"])
    for path in paths:
        probabilities = predictor.predict(path)
        detected = [name for name, value in probabilities.items() if value > predictor.threshold]
        writer.writerow([path, *(f"{probabilities[name]:.6f}" for name in predictor.classes), "|".join(detected)])


if __name__ == "__main__":
    main()
