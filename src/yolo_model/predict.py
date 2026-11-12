
import argparse
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from yolo_model.class_colors import class_color

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_unique_images(source):
    images_by_name = {}

    for image_path in sorted(
        source.iterdir(),
        key=lambda path: (path.name.casefold(), path.name),
    ):
        if (
            image_path.is_file()
            and image_path.suffix.casefold() in IMAGE_EXTENSIONS
        ):
            key = image_path.stem.casefold()
            images_by_name.setdefault(key, image_path)

    return sorted(
        images_by_name.values(),
        key=lambda image_path: (
            image_path.name.casefold(),
            image_path.name,
        ),
    )


def draw_obb(image, polygon, class_name):
    color = class_color(class_name)

    points = polygon.astype(int).reshape((-1, 1, 2))

    # Draw only the OBB outline.
    cv2.polylines(
        image,
        [points],
        isClosed=True,
        color=color,
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    # Draw class name only.
    label = class_name

    image_height, image_width = image.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.60
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    corners = points[:, 0, :]
    anchor = corners[(corners[:, 0] + corners[:, 1]).argmin()]
    text_x = min(max(int(anchor[0]), 0), max(image_width - text_width, 0))
    text_y = int(anchor[1]) - 8

    # If there is no room above the box, put the label below it.
    if text_y - text_height - baseline < 0:
        text_y = min(
            image_height - 1,
            int(anchor[1]) + text_height + baseline + 5,
        )

    cv2.putText(
        image,
        label,
        (text_x, text_y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO OBB predictions on an image or image folder."
    )
    parser.add_argument("weights", type=Path, help="Path to trained OBB weights")
    parser.add_argument("source", type=Path, help="Image file or folder of images")
    parser.add_argument(
        "confidence",
        nargs="?",
        type=float,
        default=0.001,
        help="Confidence threshold from 0 to 1 (default: 0.001)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output folder (default: a new run-specific folder under predictions)",
    )
    args = parser.parse_args()

    weights = args.weights
    source = args.source
    confidence = args.confidence

    if not 0.0 <= confidence <= 1.0:
        parser.error("confidence must be between 0 and 1")
    if not weights.is_file():
        parser.error(f"model file not found: {weights}")
    if not source.exists():
        parser.error(f"source not found: {source}")

    print()
    print("Loading YOLO11m-OBB model...")
    print(f"Weights: {weights}")
    print("Task: OBB")
    print(f"Confidence: {confidence}")
    print()

    # Explicitly load the model as OBB.
    model = YOLO(str(weights), task="obb")

    # IMPORTANT:
    # Use the class names stored inside the trained model.
    # This prevents a second manually-maintained class list
    # from becoming inconsistent with the model.
    model_names = model.names

    print(f"Model classes: {len(model_names)}")

    if source.is_file():
        image_files = [source]
    else:
        image_files = find_unique_images(source)

    if not image_files:
        parser.error(f"no supported images found in: {source}")

    print(f"Unique images found: {len(image_files)}")
    print()

    source_name = source.stem if source.is_file() else source.name
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        args.output
        if args.output is not None
        else Path("predictions") / f"{source_name}_{weights.stem}_{run_id}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_detections = 0
    images_processed = 0

    for image_path in image_files:

        print(f"Processing: {image_path.name}")

        image = cv2.imread(str(image_path))

        if image is None:

            print(
                f"WARNING: Could not read "
                f"{image_path}"
            )

            continue

        results = model.predict(
            source=str(image_path),

            # Explicit OBB inference.
            task="obb",

            # User can change this from PowerShell.
            conf=confidence,

            imgsz=640,
            device="cpu",
            verbose=False,
        )

        images_processed += 1

        result = results[0]
        detection_count = 0

        if (
            result.obb is not None
            and len(result.obb) > 0
        ):

            # YOLO11-OBB gives four corner points.
            polygons = (
                result.obb.xyxyxyxy
                .cpu()
                .numpy()
            )

            class_ids = (
                result.obb.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            for polygon, class_id in zip(
                polygons,
                class_ids,
            ):

                class_id = int(class_id)

                if class_id not in model_names:

                    print(
                        f"WARNING: Unknown class ID "
                        f"{class_id}"
                    )

                    continue

                class_name = model_names[class_id]

                draw_obb(
                    image=image,
                    polygon=polygon,
                    class_name=class_name,
                )

                detection_count += 1

        else:

            print("Model detections: 0")

        output_path = output_dir / image_path.name

        success = cv2.imwrite(
            str(output_path),
            image,
        )

        if success:

            print(
                f"Detections drawn: "
                f"{detection_count}"
            )

            print(
                f"Result saved to: "
                f"{output_path}"
            )

        else:

            print(
                f"ERROR: Could not save "
                f"{output_path}"
            )

        print()

        total_detections += detection_count

    print("=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)
    print(f"Unique images found: {len(image_files)}")
    print(f"Images processed: {images_processed}")
    print(f"Total detections: {total_detections}")
    print(f"Output directory: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
