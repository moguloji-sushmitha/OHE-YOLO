import math
import warnings
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.utils import check_det_dataset

from yolo_model.class_colors import DEFECT_CLASSES, class_color


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "configs" / "data.yaml"
    dataset = check_det_dataset(str(data_path), autodownload=False)
    class_names = dataset["names"]
    defect_ids = {
        class_id
        for class_id, class_name in class_names.items()
        if class_name in DEFECT_CLASSES
    }
    if not defect_ids:
        raise ValueError("No defect classes are configured in the dataset YAML.")

    label_directory = project_root / "dataset" / "labels" / "train"
    image_directory = project_root / "dataset" / "images" / "train"
    image_path = None
    label_path = None
    for candidate_label in sorted(label_directory.glob("*.txt")):
        lines = candidate_label.read_text(encoding="utf-8-sig").splitlines()
        if any(
            (fields := line.split()) and int(fields[0]) in defect_ids
            for line in lines
        ):
            candidate_image = image_directory / f"{candidate_label.stem}.jpg"
            if candidate_image.is_file():
                image_path = candidate_image
                label_path = candidate_label
                break

    if image_path is None or label_path is None:
        raise FileNotFoundError(
            "No training image with a labeled defect class was found."
        )
    output_path = (
        project_root / "predictions" / f"ground_truth_{image_path.stem}.jpg"
    )

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file does not exist: {label_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Could not read image file: {image_path}")

    height, width = image.shape[:2]
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        try:
            if len(fields) != 9:
                raise ValueError("expected a class ID and eight coordinates")

            class_id = int(fields[0])
            if class_id not in class_names:
                raise ValueError(f"class ID {class_id} is outside the mapping")

            coordinates = [float(value) for value in fields[1:]]
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError("coordinates must be finite numbers")
        except ValueError as error:
            warnings.warn(
                f"Skipping malformed label line {line_number}: {error}",
                UserWarning,
                stacklevel=1,
            )
            continue

        polygon = np.array(
            [
                [
                    min(max(round(coordinates[index] * width), 0), width - 1),
                    min(max(round(coordinates[index + 1] * height), 0), height - 1),
                ]
                for index in range(0, 8, 2)
            ],
            dtype=np.int32,
        )
        cv2.polylines(
            image,
            [polygon],
            isClosed=True,
            color=class_color(class_names[class_id]),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        name = class_names[class_id]
        text_size, baseline = cv2.getTextSize(
            name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        anchor = polygon[np.argmin(polygon[:, 0] + polygon[:, 1])]
        text_x = min(
            int(anchor[0]),
            max(width - text_size[0], 0),
        )
        text_y = int(anchor[1]) - 6
        if text_y - text_size[1] - baseline < 0:
            text_y = min(
                int(anchor[1]) + text_size[1] + baseline + 6,
                height - 1,
            )
        cv2.putText(
            image,
            name,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            class_color(name),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not save visualization to: {output_path}")
    print(output_path)

if __name__ == "__main__":
    main()
