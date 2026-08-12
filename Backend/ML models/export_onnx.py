import sys
from pathlib import Path
from ultralytics import YOLO

WEIGHTS_DIR = Path(__file__).parent / "weights"
PT_WEIGHTS_PATH = WEIGHTS_DIR / "plant_leaf_detection.pt"
ONNX_WEIGHTS_PATH = WEIGHTS_DIR / "plant_leaf_detection.onnx"


def export_to_onnx():
    """Export the existing PyTorch YOLO model (.pt) to ONNX format."""
    if not PT_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"PyTorch model weights not found at: {PT_WEIGHTS_PATH}")

    print(f"Loading PyTorch YOLO model from: {PT_WEIGHTS_PATH}")
    model = YOLO(str(PT_WEIGHTS_PATH))

    print(f"Exporting model to ONNX format...")
    # Ultralytics model.export() returns the exported filename/path
    exported_path = model.export(format="onnx", dynamic=False)
    
    # Ensure exported path is at WEIGHTS_DIR / "plant_leaf_detection.onnx"
    exported_file = Path(exported_path)
    if exported_file.exists() and exported_file != ONNX_WEIGHTS_PATH:
        if ONNX_WEIGHTS_PATH.exists():
            ONNX_WEIGHTS_PATH.unlink()
        exported_file.rename(ONNX_WEIGHTS_PATH)

    print(f"\nONNX Model Exported Successfully!")
    print(f"ONNX Model File Path: {ONNX_WEIGHTS_PATH}")
    print(f"File Size: {ONNX_WEIGHTS_PATH.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    export_to_onnx()
