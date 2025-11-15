from pathlib import Path

from shared import run_cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "notebooks/models/best_fp32.pt"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "notebooks/tcc-1/valid/images"
DESCRIPTION = "Ferramenta de benchmark para modelos YOLO ou ONNX."


def run_benchmark() -> None:
    run_cli(
        description=DESCRIPTION,
        default_model_format="yolo",
        default_model_path=DEFAULT_MODEL_PATH,
        default_image_dir=DEFAULT_IMAGE_DIR,
    )


if __name__ == "__main__":
    run_benchmark()