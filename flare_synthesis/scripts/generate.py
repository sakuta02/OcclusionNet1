import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flare_synthesis.config import GenerationConfig
from flare_synthesis.generate import run_generation


def main():
    parser = argparse.ArgumentParser(description="Синтез кадров с бликами поверх фоновых снимков")
    parser.add_argument("--background-dir", required=True, help="кадры вождения, на которые кладётся блик")
    parser.add_argument("--flare-dir", required=True, help="FlareX Flare2D/input")
    parser.add_argument("--light-dir", required=True, help="FlareX Flare2D/gt — источники света")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("-n", "--n-samples", type=int, default=1000)
    parser.add_argument("--img-size", type=int, default=720)
    parser.add_argument("--crop-size", type=int, default=512, help="0 — не кропать")
    parser.add_argument("--archive", action="store_true", help="дополнительно упаковать результат в zip")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = GenerationConfig(
        background_dir=Path(args.background_dir),
        flare_dir=Path(args.flare_dir),
        light_dir=Path(args.light_dir),
        output_dir=Path(args.output_dir),
        n_samples=args.n_samples,
        img_size=args.img_size,
        crop_size=args.crop_size or None,
        archive=args.archive,
        device=args.device,
    )
    result = run_generation(config)
    print(f"Готово: {result.n_generated} кадров в {result.output_dir}")
    if result.archive_path:
        print(f"Архив: {result.archive_path}")


if __name__ == "__main__":
    main()
