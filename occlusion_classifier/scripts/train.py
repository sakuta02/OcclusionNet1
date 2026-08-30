import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from occlusion_classifier.config import TrainingConfig
from occlusion_classifier.training import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    result = run_training(TrainingConfig.from_mapping(values))
    print({"best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss, "macro": result.macro_metrics})


if __name__ == "__main__":
    main()
