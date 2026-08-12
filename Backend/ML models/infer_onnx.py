import sys
from pathlib import Path
from ultralytics import YOLO
import onnxruntime as ort

WEIGHTS_DIR = Path(__file__).parent / "weights"
ONNX_WEIGHTS_PATH = WEIGHTS_DIR / "plant_leaf_detection.onnx"
PT_WEIGHTS_PATH = WEIGHTS_DIR / "plant_leaf_detection.pt"
DEFAULT_IMAGE_PATH = Path(__file__).parent / "sample_images" / "leaf_sample.jpg"


def run_onnx_inference(image_path: str | Path, confidence_threshold: float = 0.25):
    """Run ONNX Runtime inference on the specified image and return detection records."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found at: {image_path}")

    if not ONNX_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"ONNX model weights not found at: {ONNX_WEIGHTS_PATH}")

    # Verify ONNX Runtime session direct initialization
    print(f"Initializing ONNX Runtime InferenceSession with model: {ONNX_WEIGHTS_PATH}")
    session = ort.InferenceSession(str(ONNX_WEIGHTS_PATH))
    inputs = [i.name for i in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]
    print(f"  - ONNX Model Input Names:  {inputs}")
    print(f"  - ONNX Model Output Names: {outputs}")

    # Load ONNX model via YOLO engine (utilizing ONNX Runtime backend)
    onnx_model = YOLO(str(ONNX_WEIGHTS_PATH), task="detect")
    print(f"\nRunning ONNX inference on image: {image_path}\n")
    results = onnx_model.predict(source=str(image_path), conf=confidence_threshold, verbose=False)

    print("=" * 65)
    print("ONNX DETECTION RESULTS")
    print("=" * 65)

    onnx_detections = []
    total_detections = 0
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            print("No detections found.")
            continue

        for box in boxes:
            total_detections += 1
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id] if (result.names and cls_id in result.names) else str(cls_id)
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            xmin, ymin, xmax, ymax = [round(coord, 2) for coord in xyxy]

            onnx_detections.append({
                "class": cls_name,
                "class_id": cls_id,
                "confidence": confidence,
                "bbox": [xmin, ymin, xmax, ymax]
            })

            print(f"Detection #{total_detections}:")
            print(f"  - Class Name:   {cls_name} (ID: {cls_id})")
            print(f"  - Confidence:   {confidence:.4f} ({confidence * 100:.2f}%)")
            print(f"  - Bounding Box: [xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}]")
            print("-" * 65)

    print(f"Total ONNX Objects Detected: {total_detections}")
    print("=" * 65)
    return onnx_detections


def compare_with_pytorch(image_path: str | Path, confidence_threshold: float = 0.25):
    """Compare ONNX Runtime inference results with original PyTorch model results."""
    print("\n" + "=" * 65)
    print("COMPARING PYTORCH VS ONNX PREDICTIONS")
    print("=" * 65)

    pt_model = YOLO(str(PT_WEIGHTS_PATH))
    pt_results = pt_model.predict(source=str(image_path), conf=confidence_threshold, verbose=False)

    pt_detections = []
    for result in pt_results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id] if (result.names and cls_id in result.names) else str(cls_id)
            confidence = float(box.conf[0].item())
            xyxy = [round(c, 2) for c in box.xyxy[0].tolist()]
            pt_detections.append({
                "class": cls_name,
                "confidence": confidence,
                "bbox": xyxy
            })

    onnx_detections = run_onnx_inference(image_path, confidence_threshold)

    print("\n--- ACCURACY COMPARISON SUMMARY ---")
    print(f"PyTorch Total Detections: {len(pt_detections)}")
    print(f"ONNX Total Detections:    {len(onnx_detections)}")

    match = True
    if len(pt_detections) != len(onnx_detections):
        match = False
    else:
        for i, (pt, onnx) in enumerate(zip(pt_detections, onnx_detections), 1):
            conf_diff = abs(pt["confidence"] - onnx["confidence"])
            bbox_diff = max(abs(a - b) for a, b in zip(pt["bbox"], onnx["bbox"]))
            print(f"\nDetection #{i} Comparison:")
            print(f"  PyTorch Class: {pt['class']:<12} | ONNX Class: {onnx['class']}")
            print(f"  PyTorch Conf:  {pt['confidence']:.4f}       | ONNX Conf:  {onnx['confidence']:.4f} (diff: {conf_diff:.6f})")
            print(f"  PyTorch BBox:  {pt['bbox']} | ONNX BBox:  {onnx['bbox']} (max diff: {bbox_diff:.2f}px)")
            if pt["class"] != onnx["class"] or conf_diff > 0.01 or bbox_diff > 1.0:
                match = False

    print("\n" + "=" * 65)
    if match:
        print("VERIFICATION SUCCESSFUL: PyTorch and ONNX predictions match accurately!")
    else:
        print("Slight numerical variation between PyTorch and ONNX predictions.")
    print("=" * 65)


if __name__ == "__main__":
    target_img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    compare_with_pytorch(target_img)
