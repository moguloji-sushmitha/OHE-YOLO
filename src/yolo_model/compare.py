from pathlib import Path
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[2]

    weights = (
        project_root
        / "runs"
        / "obb"
        / "runs"
        / "obb"
        / "ohe_all_tasks_v1"
        / "weights"
        / "best.pt"
    )

    image_path = (
        project_root
        / "dataset"
        / "images"
        / "train"
        / "frame_000763.jpg"
    )

    label_path = (
        project_root
        / "dataset"
        / "labels"
        / "train"
        / "frame_000763.txt"
    )

    if not weights.is_file():
        raise FileNotFoundError(f"Model not found: {weights}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not label_path.is_file():
        raise FileNotFoundError(f"Label not found: {label_path}")

    # -----------------------------
    # Ground truth
    # -----------------------------
    ground_truth = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()

        if len(fields) != 9:
            continue

        class_id = int(fields[0])
        ground_truth.append(class_id)

    # -----------------------------
    # Model
    # -----------------------------
    print("Loading model...")
    model = YOLO(str(weights), task="obb")

    print()
    print("=" * 60)
    print("GROUND TRUTH vs PREDICTION")
    print("=" * 60)
    print(f"Image       : {image_path.name}")
    print(f"Ground truth: {len(ground_truth)} objects")
    print("=" * 60)

    # -----------------------------
    # Prediction
    # -----------------------------
    results = model.predict(
        source=str(image_path),
        task="obb",
        conf=0.001,
        imgsz=640,
        device="cpu",
        verbose=False,
    )

    result = results[0]

    if result.obb is None:
        print("Prediction: 0 objects")
        return

    predicted_classes = result.obb.cls.cpu().numpy().astype(int)
    confidences = result.obb.conf.cpu().numpy()

    print(f"Prediction  : {len(predicted_classes)} objects")
    print("=" * 60)

    print()
    print("PREDICTIONS:")
    print("-" * 60)

    for class_id, confidence in zip(predicted_classes, confidences):
        class_name = model.names[int(class_id)]
        print(f"{class_name:<40} {confidence:.4f}")

    print()
    print("=" * 60)

    # Ground-truth class counts
    gt_counts = {}

    for class_id in ground_truth:
        gt_counts[class_id] = gt_counts.get(class_id, 0) + 1

    # Prediction class counts
    pred_counts = {}

    for class_id in predicted_classes:
        class_id = int(class_id)
        pred_counts[class_id] = pred_counts.get(class_id, 0) + 1

    print("GROUND TRUTH CLASS COUNTS")
    print("-" * 60)

    for class_id, count in sorted(gt_counts.items()):
        print(
            f"{class_id:>3}  "
            f"{model.names[class_id]:<40} "
            f"{count}"
        )

    print()
    print("PREDICTION CLASS COUNTS")
    print("-" * 60)

    for class_id, count in sorted(pred_counts.items()):
        print(
            f"{class_id:>3}  "
            f"{model.names[class_id]:<40} "
            f"{count}"
        )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Ground truth objects : {len(ground_truth)}")
    print(f"Predicted objects    : {len(predicted_classes)}")
    print(
        f"Detection difference : "
        f"{len(predicted_classes) - len(ground_truth):+d}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()