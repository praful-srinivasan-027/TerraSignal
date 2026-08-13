import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

MODEL_REPO_ID = "foduucom/plant-leaf-detection-and-classification"
WEIGHTS_DIR = Path(__file__).parent / "weights"
LOCAL_WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
DEFAULT_IMAGE_PATH = Path(__file__).parent / "sample_images" / "000008.jpg"


def get_model_weights() -> Path:
    """Ensure pretrained weights are downloaded into local weights directory."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if LOCAL_WEIGHTS_PATH.exists():
        print(f"Using cached model weights at: {LOCAL_WEIGHTS_PATH}")
        return LOCAL_WEIGHTS_PATH

    print(f"Downloading model weights from HuggingFace ({MODEL_REPO_ID})...")
    filenames_to_try = ["model.pt", "best.pt", "pytorch_model.bin"]
    
    for filename in filenames_to_try:
        try:
            downloaded_file = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=filename,
                local_dir=str(WEIGHTS_DIR),
            )
            target_path = Path(downloaded_file)
            if target_path.exists():
                if target_path != LOCAL_WEIGHTS_PATH:
                    target_path.rename(LOCAL_WEIGHTS_PATH)
                print(f"Successfully downloaded weights to: {LOCAL_WEIGHTS_PATH}")
                return LOCAL_WEIGHTS_PATH
        except Exception:
            continue

    # Fallback to YOLO directly downloading from repo
    print("Downloading directly via Ultralytics YOLO hub loader...")
    model = YOLO(MODEL_REPO_ID)
    model.save(str(LOCAL_WEIGHTS_PATH))
    return LOCAL_WEIGHTS_PATH


def run_inference(image_path: str | Path, confidence_threshold: float = 0.25):
    """Load the YOLO model and run inference on the specified image."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found at: {image_path}")

    weights_path = get_model_weights()
    print(f"Loading YOLO model from: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"Running inference on image: {image_path}\n")
    results = model.predict(source=str(image_path), conf=confidence_threshold, verbose=False)

    print("=" * 65)
    print("DETECTION RESULTS")
    print("=" * 65)

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

            print(f"Detection #{total_detections}:")
            print(f"  - Class Name:   {cls_name} (ID: {cls_id})")
            print(f"  - Confidence:   {confidence:.4f} ({confidence * 100:.2f}%)")
            print(f"  - Bounding Box: [xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}]")
            print("-" * 65)

        # Save annotated image with visual bounding boxes
        output_dir = Path(__file__).parent / "output_images"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"detected_{image_path.name}"
        
        # Plot returns an annotated BGR numpy array
        annotated_frame = result.plot()
        import cv2
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"\nSaved visual bounding boxes image to: {output_path}")

        # Display window using OpenCV
        try:
            print("Opening OpenCV window with visual bounding boxes (press any key on window to close)...")
            cv2.imshow("Plant Leaf Detections", annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Note: Could not open OpenCV GUI display window ({e}). Image saved to {output_path}")

    print(f"Total Objects Detected: {total_detections}")
    print("=" * 65)


if __name__ == "__main__":
    target_img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    run_inference(target_img)
