import argparse
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from occlusion_score.config import TrainingConfig
from occlusion_score.training import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    result = run_training(TrainingConfig.from_mapping(values))
    print({"status": result.status, "teacher": result.teacher_metrics, "student": result.student_metrics})


if __name__ == "__main__":
    main()
