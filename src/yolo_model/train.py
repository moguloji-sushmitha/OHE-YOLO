
"""
Train YOLO11m-OBB for OHE component/defect detection.

Example:
    python src/yolo_model/train.py

CPU testing:
    python src/yolo_model/train.py --epochs 1 --imgsz 640 --batch 1

Training:
    python src/yolo_model/train.py --epochs 100 --imgsz 640 --batch 1
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

from yolo_model.class_colors import class_color


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _image_paths(source: str | Path | list[str | Path]) -> list[Path]:
    sources = source if isinstance(source, list) else [source]
    paths = []
    for item in sources:
        path = Path(item)
        if path.is_dir():
            paths.extend(
                image
                for image in sorted(path.iterdir())
                if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
        else:
            raise FileNotFoundError(f"Validation image source not found: {path}")
    return paths


def _label_directory(image_directory: Path) -> Path:
    parts = list(image_directory.parts)
    image_index = next(
        (index for index in range(len(parts) - 1, -1, -1) if parts[index] == "images"),
        None,
    )
    if image_index is None:
        raise ValueError(
            f"Cannot locate the images directory in validation path: {image_directory}"
        )
    parts[image_index] = "labels"
    return Path(*parts)


def _validate_labeled_splits(data_path: Path) -> dict:
    dataset = check_det_dataset(str(data_path), autodownload=False)
    names = dataset["names"]
    for split in ("train", "val"):
        images = _image_paths(dataset[split])
        if not images:
            raise ValueError(f"The dataset {split} split contains no images")
        for image_path in images:
            label_path = _label_directory(image_path.parent) / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"{split} image has no matching ground-truth label: {label_path}"
                )
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
            ):
                fields = line.split()
                if len(fields) != 9:
                    raise ValueError(
                        f"{label_path}:{line_number}: expected a class ID and eight coordinates"
                    )
                class_id = int(fields[0])
                if class_id not in names:
                    raise ValueError(
                        f"{label_path}:{line_number}: unknown class ID {class_id}"
                    )
                coordinates = [float(value) for value in fields[1:]]
                if not all(math.isfinite(value) for value in coordinates):
                    raise ValueError(
                        f"{label_path}:{line_number}: coordinates must be finite"
                    )
                if any(value < -0.01 or value > 1.01 for value in coordinates):
                    raise ValueError(
                        f"{label_path}:{line_number}: coordinates must be normalized"
                    )
        print(f"Validated {split} labels: {len(images)} images")
    return dataset


def _draw_ground_truth(image: np.ndarray, label_path: Path, names) -> np.ndarray:
    height, width = image.shape[:2]
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(
                f"{label_path}:{line_number}: expected a class ID and eight coordinates"
            )
        class_id = int(fields[0])
        if class_id not in names:
            raise ValueError(f"{label_path}:{line_number}: unknown class ID {class_id}")
        coordinates = [float(value) for value in fields[1:]]
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{label_path}:{line_number}: coordinates must be finite")

        polygon = np.asarray(
            [
                [
                    min(max(round(coordinates[index] * width), 0), width - 1),
                    min(max(round(coordinates[index + 1] * height), 0), height - 1),
                ]
                for index in range(0, 8, 2)
            ],
            dtype=np.int32,
        )
        label = str(names[class_id])
        color = class_color(label)
        cv2.polylines(
            image,
            [polygon.reshape((-1, 1, 2))],
            True,
            color,
            2,
            cv2.LINE_AA,
        )
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        anchor = polygon[np.argmin(polygon[:, 0] + polygon[:, 1])]
        x = min(int(anchor[0]), max(0, width - text_width))
        text_y = int(anchor[1]) - 6
        if text_y - text_height - baseline < 0:
            text_y = min(
                height - 1,
                int(anchor[1]) + text_height + baseline + 6,
            )
        cv2.putText(
            image,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return image


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Train YOLO11m-OBB on the OHE dataset"
    )

    parser.add_argument(
        "--data",
        default=str(project_root / "configs" / "data.yaml"),
        help="Path to OBB dataset YAML",
    )

    parser.add_argument(
        "--model",
        default="yolo11m-obb.pt",
        help="YOLO OBB model",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size. 640 is recommended for CPU.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size. Use 1 for CPU",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or GPU number such as 0",
    )

    parser.add_argument(
        "--name",
        default="ohe_obb_current",
        help="Training run name",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for saved validation predictions",
    )

    args = parser.parse_args()

    data_path = Path(args.data).resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}"
        )
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    dataset = _validate_labeled_splits(data_path)

    run_directory = project_root / "runs" / "obb" / args.name
    results_directory = project_root / "predictions" / args.name
    print("=" * 60)
    print("OHE YOLO11m-OBB TRAINING")
    print("=" * 60)
    print(f"Model       : {args.model}")
    print(f"Task        : OBB")
    print(f"Dataset     : {data_path}")
    print(f"Image size  : {args.imgsz}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch       : {args.batch}")
    print(f"Device      : {args.device}")
    print("=" * 60)

    # Explicitly load the OBB model.
    model = YOLO(args.model, task="obb")

    model.train(
        data=str(data_path),

        # Explicitly force Oriented Bounding Box training.
        task="obb",

        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,

        # Keep OBB runs in a dedicated directory.
        project=str(project_root / "runs" / "obb"),
        name=args.name,

        # Training settings.
        patience=30,
        pretrained=True,

        # OBB-compatible augmentation.
        fliplr=0.5,
        flipud=0.0,
        degrees=10,
        translate=0.1,
        scale=0.5,

        # Color augmentation.
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Spatial augmentation.
        mosaic=1.0,
        mixup=0.0,

        # CPU-friendly setting.
        workers=0,

        # No graphs required.
        plots=False,
    )

    run_directory = Path(model.trainer.save_dir)
    best_weights = run_directory / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Best model weights were not created: {best_weights}")

    best_model = YOLO(str(best_weights), task="obb")
    metrics = best_model.val(
        data=str(data_path),
        task="obb",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        plots=False,
        project=str(run_directory),
        name="final_validation",
        exist_ok=True,
    )

    validation_source = dataset["val"]
    validation_images = _image_paths(validation_source)
    if not validation_images:
        raise FileNotFoundError(f"No validation images found in: {validation_source}")

    prediction_directory = results_directory / "predictions"
    prediction_count = 0
    for image_path in validation_images:
        label_path = _label_directory(image_path.parent) / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(
                f"Validation image has no matching ground-truth label: {label_path}"
            )

        prediction = best_model.predict(
            source=str(image_path),
            task="obb",
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            verbose=False,
        )[0]
        annotated = cv2.imread(str(image_path))
        if annotated is None:
            raise OSError(f"Could not read validation image: {image_path}")
        _draw_ground_truth(annotated, label_path, best_model.names)
        if prediction.obb is not None:
            polygons = prediction.obb.xyxyxyxy.cpu().numpy()
            class_ids = prediction.obb.cls.cpu().numpy().astype(int)
            confidences = prediction.obb.conf.cpu().numpy()
            for polygon, class_id, confidence in zip(
                polygons, class_ids, confidences
            ):
                points = np.rint(polygon).astype(np.int32).reshape((-1, 1, 2))
                class_name = str(best_model.names[class_id])
                color = class_color(class_name)
                cv2.polylines(
                    annotated, [points], True, color, 3, cv2.LINE_AA
                )
                label = f"{class_name} {confidence:.2f}"
                corners = points[:, 0, :]
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                anchor = corners[np.argmin(corners[:, 0] + corners[:, 1])]
                x = min(
                    max(0, int(anchor[0])),
                    max(0, annotated.shape[1] - text_width),
                )
                text_y = int(anchor[1]) - 6
                if text_y - text_height - baseline < 0:
                    text_y = min(
                        annotated.shape[0] - 1,
                        int(anchor[1]) + text_height + baseline + 6,
                    )
                cv2.putText(
                    annotated,
                    label,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        prediction_path = prediction_directory / image_path.name
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(prediction_path), annotated):
            raise OSError(f"Could not save prediction image: {prediction_path}")
        prediction_count += len(prediction.obb) if prediction.obb is not None else 0

    metrics_path = results_directory / "metrics.txt"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        "\n".join(
            (
                f"Precision: {metrics.box.mp:.6f}",
                f"Recall: {metrics.box.mr:.6f}",
                f"mAP50: {metrics.box.map50:.6f}",
                f"mAP50-95: {metrics.box.map:.6f}",
                f"Prediction confidence threshold: {args.conf:.3f}",
                f"Validation images: {len(validation_images)}",
                f"Predictions at threshold: {prediction_count}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print("Task        : OBB")
    print(f"Output      : {run_directory}")
    print(f"Best model  : {best_weights}")
    print(f"Results     : {results_directory}")
    print(
        f"Combined validation images: {len(validation_images)} "
        f"(ground truth + predictions)"
    )
    print(f"Model predictions: {prediction_count} at confidence {args.conf:.2f}")
    print(f"mAP50       : {metrics.box.map50:.4f}")
    print(f"mAP50-95    : {metrics.box.map:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
