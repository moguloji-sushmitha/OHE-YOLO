
import argparse
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from yolo_model.class_colors import class_color

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpeg", ".mpg"}


def find_media_files(source: Path) -> list[Path]:
    supported_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    return sorted(
        (
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.casefold() in supported_extensions
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )


def draw_obb(image, polygon, class_name, confidence):
    color = class_color(class_name)
    points = polygon.astype(int).reshape((-1, 1, 2))
    image_height, image_width = image.shape[:2]
    scale = max(min(image_width, image_height) / 640, 1.0)
    line_thickness = max(2, round(2 * scale))

    cv2.polylines(
        image,
        [points],
        isClosed=True,
        color=color,
        thickness=line_thickness,
        lineType=cv2.LINE_AA,
    )

    label = f"{class_name} {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 * scale
    thickness = max(1, round(1.5 * scale))

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    corners = points[:, 0, :]
    anchor = corners[(corners[:, 0] + corners[:, 1]).argmin()]
    label_x = min(max(int(anchor[0]), 0), max(image_width - text_width, 0))
    label_y = int(anchor[1]) - baseline - 2
    if label_y - text_height < 0:
        label_y = min(int(anchor[1]) + text_height + baseline + 2, image_height - baseline)

    text_origin = (label_x, label_y)
    cv2.putText(
        image,
        label,
        text_origin,
        font,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        text_origin,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def annotate_result(image, result, model_names) -> int:
    if result.obb is None or len(result.obb) == 0:
        return 0

    polygons = result.obb.xyxyxyxy.cpu().numpy()
    class_ids = result.obb.cls.cpu().numpy().astype(int)
    confidences = result.obb.conf.cpu().numpy()
    detection_count = 0
    for polygon, class_id, confidence in zip(polygons, class_ids, confidences):
        class_name = model_names.get(int(class_id))
        if class_name is None:
            print(f"WARNING: Unknown class ID {class_id}")
            continue
        draw_obb(image, polygon, class_name, float(confidence))
        detection_count += 1
    return detection_count


def predict_frame(model, image, model_names, confidence) -> int:
    result = model.predict(
        source=image,
        task="obb",
        conf=confidence,
        imgsz=640,
        device="cpu",
        verbose=False,
    )[0]
    return annotate_result(image, result, model_names)


def predict_video(
    model, video_path, output_path, model_names, confidence
) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise OSError(f"Could not read video dimensions: {video_path}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise OSError(f"Could not create prediction video: {output_path}")

    frames_processed = 0
    total_detections = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            total_detections += predict_frame(
                model, frame, model_names, confidence
            )
            writer.write(frame)
            frames_processed += 1
    finally:
        capture.release()
        writer.release()

    if frames_processed == 0:
        raise OSError(f"Video contains no readable frames: {video_path}")
    return frames_processed, total_detections


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO OBB predictions on an image, video, or folder."
    )
    parser.add_argument("weights", type=Path, help="Path to trained OBB weights")
    parser.add_argument(
        "source", type=Path, help="Image/video file or folder of media files"
    )
    parser.add_argument(
        "confidence",
        nargs="?",
        type=float,
        default=0.25,
        help="Confidence threshold from 0 to 1 (default: 0.25)",
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
        media_files = [source]
    else:
        media_files = find_media_files(source)

    supported_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    unsupported_files = [
        path
        for path in media_files
        if path.suffix.casefold() not in supported_extensions
    ]
    if source.is_file() and unsupported_files:
        parser.error(
            f"unsupported media file: {unsupported_files[0]}; supported formats "
            f"are images {sorted(IMAGE_EXTENSIONS)} and videos "
            f"{sorted(VIDEO_EXTENSIONS)}"
        )
    if not media_files:
        parser.error(f"no supported images or videos found in: {source}")

    print(f"Media files found: {len(media_files)}")
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
    media_processed = 0
    frames_processed = 0

    for media_path in media_files:
        print(f"Processing: {media_path.name}")

        if media_path.suffix.casefold() in VIDEO_EXTENSIONS:
            output_path = output_dir / f"{media_path.stem}_predicted.mp4"
            video_frames, detections = predict_video(
                model,
                media_path,
                output_path,
                model_names,
                confidence,
            )
            frames_processed += video_frames
            total_detections += detections
            media_processed += 1
            print(f"Frames processed: {video_frames}")
            print(f"Detections drawn: {detections}")
        else:
            try:
                image = cv2.imread(str(media_path))
            except cv2.error as error:
                print(f"WARNING: Could not read image {media_path}: {error}")
                continue
            if image is None:
                print(f"WARNING: Could not read image {media_path}")
                continue

            output_path = output_dir / media_path.name
            detection_count = predict_frame(
                model, image, model_names, confidence
            )
            if not cv2.imwrite(str(output_path), image):
                raise OSError(f"Could not save prediction image: {output_path}")
            total_detections += detection_count
            media_processed += 1
            frames_processed += 1
            print(f"Detections drawn: {detection_count}")

        print(f"Result saved to: {output_path}")
        print()

    print("=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)
    print(f"Media files found: {len(media_files)}")
    print(f"Media files processed: {media_processed}")
    print(f"Frames processed: {frames_processed}")
    print(f"Total detections: {total_detections}")
    print(f"Output directory: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
