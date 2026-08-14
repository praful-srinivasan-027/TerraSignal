import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO

# Default directory definitions relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_PATH = BASE_DIR / "weights" / "best.pt"
DEFAULT_IMAGES_DIR = BASE_DIR / "sample_images"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output_images" / "marked_false_positives"


def resolve_model_weights(weights_path: Path | str) -> Path:
    """Resolve model weights path, falling back to alternative weights if default is missing."""
    target_path = Path(weights_path)
    if target_path.exists():
        return target_path

    # Try fallback filenames in weights directory
    weights_dir = BASE_DIR / "weights"
    fallbacks = [
        weights_dir / "best.pt",
        weights_dir / "tbest.pt",
        weights_dir / "plant_leaf_detection.pt",
    ]
    for fb in fallbacks:
        if fb.exists():
            print(f"[INFO] Specified weights '{weights_path}' not found. Using fallback: {fb}", flush=True)
            return fb

    raise FileNotFoundError(f"No valid YOLO model weights found at: {weights_path}")


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes [xmin, ymin, xmax, ymax]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea == 0.0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return float(iou)


def load_yolo_labels(label_path: Path, img_width: int, img_height: int) -> List[Dict]:
    """Parse standard YOLO GT label file (class_id, x_center, y_center, width, height normalized)."""
    gt_boxes = []
    if not label_path.exists():
        return gt_boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])

            xmin = (xc - w / 2.0) * img_width
            ymin = (yc - h / 2.0) * img_height
            xmax = (xc + w / 2.0) * img_width
            ymax = (yc + h / 2.0) * img_height

            gt_boxes.append({
                "class_id": cls_id,
                "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)]
            })
    return gt_boxes


def auto_detect_labels_dir(images_dir: Path) -> Optional[Path]:
    """Auto-detect sibling labels directory (e.g., valid/images -> valid/labels)."""
    if images_dir.name.lower() in ("images", "img"):
        sibling_labels = images_dir.parent / "labels"
        if sibling_labels.exists() and sibling_labels.is_dir():
            return sibling_labels
    return None


def evaluate_image(
    image_path: Path,
    model: YOLO,
    tomato_class_ids: set,
    labels_dir: Optional[Path] = None,
    conf_threshold: float = 0.25,
    iou_match_threshold: float = 0.45,
    fp_threshold: int = 2,
    tomato_fp_threshold: int = 2,
) -> Dict:
    """
    Run inference on a single image and evaluate predictions for false positives,
    especially tomato leaf disease false positives.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Unable to read image at: {image_path}")

    height, width = img_bgr.shape[:2]
    results = model.predict(source=str(image_path), conf=conf_threshold, verbose=False)
    
    predictions = []
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            cls_name = results[0].names[cls_id] if (results[0].names and cls_id in results[0].names) else str(cls_id)
            conf = float(box.conf[0].item())
            xyxy = [round(c, 2) for c in box.xyxy[0].tolist()]
            is_tomato = cls_id in tomato_class_ids or "tomato" in cls_name.lower()

            predictions.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "bbox": xyxy,
                "is_tomato": is_tomato,
                "is_fp": False,
                "fp_reason": ""
            })

    # Check for GT labels
    gt_label_file = None
    if labels_dir:
        gt_label_file = labels_dir / f"{image_path.stem}.txt"

    has_gt = bool(gt_label_file and gt_label_file.exists())
    gt_boxes = load_yolo_labels(gt_label_file, width, height) if has_gt else []

    total_fps = 0
    tomato_fps = 0
    total_tps = 0
    total_fns = 0

    per_image_class_stats = {}  # {cls_id: {'gt': int, 'tp': int, 'fp': int, 'fn': int}}

    if has_gt:
        # Ground Truth Evaluation Mode
        gt_matched = [False] * len(gt_boxes)

        for gt in gt_boxes:
            cid = gt["class_id"]
            if cid not in per_image_class_stats:
                per_image_class_stats[cid] = {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
            per_image_class_stats[cid]["gt"] += 1

        for pred in predictions:
            cid = pred["class_id"]
            if cid not in per_image_class_stats:
                per_image_class_stats[cid] = {"gt": 0, "tp": 0, "fp": 0, "fn": 0}

            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if pred["class_id"] == gt["class_id"]:
                    iou = calculate_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_iou >= iou_match_threshold and best_gt_idx != -1 and not gt_matched[best_gt_idx]:
                gt_matched[best_gt_idx] = True
                pred["is_fp"] = False
                total_tps += 1
                per_image_class_stats[cid]["tp"] += 1
            else:
                pred["is_fp"] = True
                pred["fp_reason"] = f"Ground truth mismatch / No matching box (Max IoU: {best_iou:.2f})"
                total_fps += 1
                per_image_class_stats[cid]["fp"] += 1
                if pred["is_tomato"]:
                    tomato_fps += 1

        for gt_idx, matched in enumerate(gt_matched):
            if not matched:
                total_fns += 1
                cid = gt_boxes[gt_idx]["class_id"]
                per_image_class_stats[cid]["fn"] += 1
    else:
        # Heuristic / Unannotated Evaluation Mode
        n_preds = len(predictions)
        for i in range(n_preds):
            for j in range(i + 1, n_preds):
                p1, p2 = predictions[i], predictions[j]
                iou = calculate_iou(p1["bbox"], p2["bbox"])

                if p1["is_tomato"] and p2["is_tomato"]:
                    if p1["class_id"] != p2["class_id"] and iou >= 0.38:
                        lower_idx = i if p1["confidence"] < p2["confidence"] else j
                        if not predictions[lower_idx]["is_fp"]:
                            predictions[lower_idx]["is_fp"] = True
                            predictions[lower_idx]["fp_reason"] = (
                                f"Conflicting tomato prediction with {predictions[i if lower_idx==j else j]['class_name']} "
                                f"(IoU: {iou:.2f})"
                            )
                    elif p1["class_id"] == p2["class_id"] and iou >= 0.75:
                        lower_idx = i if p1["confidence"] < p2["confidence"] else j
                        if not predictions[lower_idx]["is_fp"]:
                            predictions[lower_idx]["is_fp"] = True
                            predictions[lower_idx]["fp_reason"] = f"Duplicate box for same class (IoU: {iou:.2f})"

        for pred in predictions:
            cid = pred["class_id"]
            if cid not in per_image_class_stats:
                per_image_class_stats[cid] = {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
            if pred["is_fp"]:
                total_fps += 1
                per_image_class_stats[cid]["fp"] += 1
                if pred["is_tomato"]:
                    tomato_fps += 1
            else:
                per_image_class_stats[cid]["tp"] += 1

    is_flagged = (tomato_fps >= tomato_fp_threshold) or (total_fps >= fp_threshold)

    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "total_detections": len(predictions),
        "total_gt_boxes": len(gt_boxes),
        "total_tps": total_tps,
        "total_fps": total_fps,
        "tomato_fps": tomato_fps,
        "total_fns": total_fns,
        "is_flagged": is_flagged,
        "has_ground_truth": has_gt,
        "class_stats": per_image_class_stats,
        "predictions": predictions,
    }


def draw_marked_image(image_path: Path, eval_result: Dict, output_path: Path):
    """Render bounding boxes and warning banner on flagged images and save to output directory."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return

    height, width = img_bgr.shape[:2]

    # Banner dimensions
    banner_height = 45
    banner = np.zeros((banner_height, width, 3), dtype=np.uint8)

    if eval_result["is_flagged"]:
        banner[:] = (0, 0, 180)  # Red background banner for flagged images
        status_text = f"FLAGGED: HIGH FP DETECTED | Tomato FPs: {eval_result['tomato_fps']} | Total FPs: {eval_result['total_fps']}"
    else:
        banner[:] = (40, 140, 40)  # Green background banner for normal images
        status_text = f"PASSED: Normal Detections | Total: {eval_result['total_detections']}"

    cv2.putText(banner, status_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Attach banner to top of image
    marked_img = np.vstack([banner, img_bgr])

    # Draw boxes
    for pred in eval_result["predictions"]:
        xmin, ymin, xmax, ymax = pred["bbox"]
        # Offset y coords for top banner
        ymin_b = int(ymin + banner_height)
        ymax_b = int(ymax + banner_height)
        xmin_b = int(xmin)
        xmax_b = int(xmax)

        is_fp = pred["is_fp"]
        box_color = (0, 0, 255) if is_fp else (0, 255, 0)  # Red for FP, Green for TP
        thickness = 3 if is_fp else 2

        cv2.rectangle(marked_img, (xmin_b, ymin_b), (xmax_b, ymax_b), box_color, thickness)

        label_prefix = "[FP] " if is_fp else ""
        label = f"{label_prefix}{pred['class_name']} {pred['confidence']:.2f}"

        # Draw label background
        (w_label, h_label), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin_b - 5, banner_height + 15)
        cv2.rectangle(marked_img, (xmin_b, label_ymin - h_label - 4), (xmin_b + w_label + 6, label_ymin + 2), box_color, -1)
        cv2.putText(marked_img, label, (xmin_b + 3, label_ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), marked_img)


def print_class_tree(class_id: int, class_name: str, gt: int, tp: int, fp: int, fn: int, p: float, r: float, f1: float):
    """Format and print class evaluation metrics in a tree structure."""
    print(f"\n{class_name} (Class ID: {class_id})", flush=True)
    print(f" ├── GT count  : {gt}", flush=True)
    print(f" ├── TP        : {tp}", flush=True)
    print(f" ├── FP        : {fp}", flush=True)
    print(f" ├── FN        : {fn}", flush=True)
    print(f" ├── Precision : {p:.4f}", flush=True)
    print(f" ├── Recall    : {r:.4f}", flush=True)
    print(f" └── F1        : {f1:.4f}", flush=True)


def run_evaluation(
    images_dir: Path,
    weights_path: Path,
    labels_dir: Optional[Path] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    conf_threshold: float = 0.25,
    fp_threshold: int = 2,
    tomato_fp_threshold: int = 2,
    iou_match_threshold: float = 0.45,
):
    """Main evaluation routine scanning images, identifying false positives, and producing reports."""
    resolved_weights = resolve_model_weights(weights_path)
    
    # Auto-detect labels directory if not specified
    if labels_dir is None:
        auto_labels = auto_detect_labels_dir(images_dir)
        if auto_labels:
            labels_dir = auto_labels
            print(f"[INFO] Auto-detected Ground Truth labels directory: {labels_dir}", flush=True)

    print(f"=================================================================", flush=True)
    print(f"YOLO FALSE POSITIVE EVALUATION TOOL", flush=True)
    print(f"=================================================================", flush=True)
    print(f"Model Weights:      {resolved_weights}", flush=True)
    print(f"Images Directory:   {images_dir}", flush=True)
    print(f"Labels Directory:   {labels_dir if labels_dir else 'None (Heuristic Overlap Mode)'}", flush=True)
    print(f"Output Directory:   {output_dir}", flush=True)
    print(f"Confidence Thresh:  {conf_threshold}", flush=True)
    print(f"FP Threshold:       {fp_threshold}", flush=True)
    print(f"Tomato FP Thresh:   {tomato_fp_threshold}", flush=True)
    print(f"=================================================================\n", flush=True)

    print(f"Loading YOLO model...", flush=True)
    model = YOLO(str(resolved_weights))

    names_dict = model.names if hasattr(model, "names") else {}
    tomato_class_ids = {
        cls_id for cls_id, cls_name in names_dict.items()
        if "tomato" in str(cls_name).lower()
    }
    print(f"Identified {len(tomato_class_ids)} Tomato Classes out of {len(names_dict)} total classes:", flush=True)
    for tid in sorted(tomato_class_ids):
        print(f"  - ID {tid:2d}: {names_dict[tid]}", flush=True)
    print("-" * 65, flush=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([
        p for p in Path(images_dir).glob("*")
        if p.suffix.lower() in valid_extensions
    ])

    if not image_files:
        print(f"[WARNING] No valid image files found in: {images_dir}", flush=True)
        return

    print(f"Found {len(image_files)} image(s) to evaluate.\n", flush=True)

    results_summary = []
    flagged_images = []
    total_dataset_tps = 0
    total_dataset_fps = 0
    total_dataset_fns = 0
    total_dataset_gt = 0
    total_dataset_tomato_fps = 0

    # Accumulate per-class statistics across entire dataset
    dataset_class_stats = {
        cid: {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
        for cid in names_dict.keys()
    }

    for idx, img_path in enumerate(image_files, 1):
        eval_res = evaluate_image(
            image_path=img_path,
            model=model,
            tomato_class_ids=tomato_class_ids,
            labels_dir=labels_dir,
            conf_threshold=conf_threshold,
            iou_match_threshold=iou_match_threshold,
            fp_threshold=fp_threshold,
            tomato_fp_threshold=tomato_fp_threshold,
        )

        results_summary.append(eval_res)
        total_dataset_tps += eval_res["total_tps"]
        total_dataset_fps += eval_res["total_fps"]
        total_dataset_fns += eval_res["total_fns"]
        total_dataset_gt += eval_res["total_gt_boxes"]
        total_dataset_tomato_fps += eval_res["tomato_fps"]

        for cid, stats in eval_res["class_stats"].items():
            if cid not in dataset_class_stats:
                dataset_class_stats[cid] = {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
            dataset_class_stats[cid]["gt"] += stats["gt"]
            dataset_class_stats[cid]["tp"] += stats["tp"]
            dataset_class_stats[cid]["fp"] += stats["fp"]
            dataset_class_stats[cid]["fn"] += stats["fn"]

        status_str = "[FLAGGED]" if eval_res["is_flagged"] else "[OK]"
        gt_info = f" | GT Boxes: {eval_res['total_gt_boxes']}" if eval_res["has_ground_truth"] else ""
        print(
            f"[{idx}/{len(image_files)}] {img_path.name:<25} | Status: {status_str:<9} "
            f"| Detections: {eval_res['total_detections']}{gt_info} | Tomato FPs: {eval_res['tomato_fps']} | Total FPs: {eval_res['total_fps']}",
            flush=True
        )

        if eval_res["is_flagged"]:
            flagged_images.append(eval_res)
            out_img_path = output_dir / f"flagged_{img_path.name}"
            draw_marked_image(img_path, eval_res, out_img_path)
            print(f"    --> Saved visual marked image: {out_img_path}", flush=True)

    # Compute per-class Precision, Recall, F1
    per_class_summary = {}
    for cid in sorted(dataset_class_stats.keys()):
        c_name = names_dict.get(cid, f"Class_{cid}")
        c_gt = dataset_class_stats[cid]["gt"]
        c_tp = dataset_class_stats[cid]["tp"]
        c_fp = dataset_class_stats[cid]["fp"]
        c_fn = dataset_class_stats[cid]["fn"]

        # Only include classes that appeared in GT or Predictions
        if c_gt > 0 or (c_tp + c_fp) > 0:
            p = c_tp / float(c_tp + c_fp + 1e-6) if (c_tp + c_fp) > 0 else 0.0
            r = c_tp / float(c_tp + c_fn + 1e-6) if (c_tp + c_fn) > 0 else 0.0
            f1 = 2 * (p * r) / (p + r + 1e-6) if (p + r) > 0 else 0.0

            per_class_summary[c_name] = {
                "class_id": cid,
                "class_name": c_name,
                "gt_count": c_gt,
                "tp": c_tp,
                "fp": c_fp,
                "fn": c_fn,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1_score": round(f1, 4),
            }

    # Global Metrics Calculation
    precision = total_dataset_tps / float(total_dataset_tps + total_dataset_fps + 1e-6) if (total_dataset_tps + total_dataset_fps) > 0 else 0.0
    recall = total_dataset_tps / float(total_dataset_tps + total_dataset_fns + 1e-6) if (total_dataset_tps + total_dataset_fns) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0

    # Generate JSON Report
    report_data = {
        "weights_used": str(resolved_weights),
        "total_images_evaluated": len(image_files),
        "total_flagged_images": len(flagged_images),
        "confidence_threshold": conf_threshold,
        "fp_threshold": fp_threshold,
        "tomato_fp_threshold": tomato_fp_threshold,
        "metrics": {
            "total_ground_truth_boxes": total_dataset_gt,
            "total_true_positives": total_dataset_tps,
            "total_false_positives": total_dataset_fps,
            "total_tomato_false_positives": total_dataset_tomato_fps,
            "total_false_negatives": total_dataset_fns,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
        },
        "per_class_metrics": per_class_summary,
        "flagged_images": [f["image_name"] for f in flagged_images],
        "evaluations": results_summary,
    }

    report_file = output_dir / "evaluation_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 65, flush=True)
    print("PER-CLASS EVALUATION BREAKDOWN", flush=True)
    print("=" * 65, flush=True)
    for c_name, stats in per_class_summary.items():
        print_class_tree(
            class_id=stats["class_id"],
            class_name=stats["class_name"],
            gt=stats["gt_count"],
            tp=stats["tp"],
            fp=stats["fp"],
            fn=stats["fn"],
            p=stats["precision"],
            r=stats["recall"],
            f1=stats["f1_score"],
        )

    print("\n" + "=" * 65, flush=True)
    print("OVERALL EVALUATION SUMMARY REPORT", flush=True)
    print("=" * 65, flush=True)
    print(f"Total Images Evaluated:         {len(image_files)}", flush=True)
    print(f"Total Flagged (High FP) Images: {len(flagged_images)}", flush=True)
    print(f"Total Tomato False Positives:   {total_dataset_tomato_fps}", flush=True)
    print(f"Total All False Positives:      {total_dataset_fps}", flush=True)
    if total_dataset_gt > 0:
        print(f"Total Ground Truth Boxes:       {total_dataset_gt}", flush=True)
        print(f"Overall Precision:              {precision:.4f}", flush=True)
        print(f"Overall Recall:                 {recall:.4f}", flush=True)
        print(f"Overall F1 Score:               {f1_score:.4f}", flush=True)
    print(f"Flagged Output Dir:              {output_dir}", flush=True)
    print(f"Detailed Report File:            {report_file}", flush=True)
    if flagged_images:
        print("\nFlagged Images List:", flush=True)
        for fi in flagged_images:
            print(f"  - {fi['image_name']} (Tomato FPs: {fi['tomato_fps']}, Total FPs: {fi['total_fps']})", flush=True)
    print("=" * 65, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation script for YOLO best.pt to detect and mark images with high false positives (especially tomatoes)."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing test images to evaluate (default: sample_images)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help="Path to YOLO best.pt model weights (default: weights/best.pt)",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Optional directory containing ground truth YOLO format .txt labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory to save marked false positive images and report",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions (default: 0.25)",
    )
    parser.add_argument(
        "--fp-threshold",
        type=int,
        default=2,
        help="False positive count threshold to flag an image (default: 2)",
    )
    parser.add_argument(
        "--tomato-fp-threshold",
        type=int,
        default=2,
        help="Tomato false positive threshold to flag an image (default: 2)",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.45,
        help="IoU threshold for ground truth matching (default: 0.45)",
    )

    args = parser.parse_args()

    run_evaluation(
        images_dir=args.images_dir,
        weights_path=args.weights,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        fp_threshold=args.fp_threshold,
        tomato_fp_threshold=args.tomato_fp_threshold,
        iou_match_threshold=args.iou_thresh,
    )


if __name__ == "__main__":
    main()
