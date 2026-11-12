import math
import warnings
from pathlib import Path
import cv2
import numpy as np

from yolo_model.class_colors import class_color

CLASS_NAMES = (
    "mast_bracket",
    "swivel_clip_rust",
    "catenary_dropper_clip",
    "suspension_insulator_bend",
    "contact_wire_dropper_clamp_rust",
    "mast_bracket_fitting",
    "bird_nest",
    "contact_wire",
    "catenary_ending_clamp",
    "anchor_clamp",
    "tesnion_struct",
    "guide_pulley_support_bracket",
    "bracket_tube",
    "mast_anchor_fitting",
    "catenary_suspension_clamp",
    "stay_tube_rust",
    "intermediate_pulley",
    "stay_tube",
    "swivel",
    "contact_wire_parallel_clamp",
    "parallel_clamp",
    "thread",
    "seddle_clamp",
    "stay_rod",
    "registration_tube",
    "movable_pulley_block",
    "counterweight_balance_mark_missing",
    "tension_clamp",
    "insulator",
    "suspension_insulator",
    "mast_guy_rod_fit",
    "tie_rod_rust",
    "tie_rod",
    "hook_bracket",
    "turnbuckle",
    "drop_bracket",
    "registration_tube_dropper_clamp",
    "stay_adjustor",
    "counterweight",
    "tube_cap",
    "tension_rod",
    "catenory_wire",
    "catenory_suspension_bracket_rust",
    "catenory_wire_rust",
    "insulator_broken",
    "fixed_pulley_block",
    "catenary_suspension_bracket",
    "clevis",
    "wire_rope",
    "swivel_clip",
    "mast",
    "anti_wind_clamp",
    "anti_bird_disk",
    "counterweight_misalignement",
    "dropper",
    "registration_tube_hook",
    "clamp",
    "BFB_steady_arm",
    "TT_sleeve",
    "compensating_plate",
    "registration_tube_dropper_break",
    "insulator_muddy",
    "dropper_clamp",
    "catenory_suspension_clamp_rust",
    "contact_wire_dropper_clip",
    "registration_tube_dropper",
    "earth_wire",
    "Jumper",
    "bracket_tube_rust",
    "cantiliver_assembly",
    "diagonal_stay_wire",
    "vacume_circuit_breaker",
    "incoming_OHE_line",
)
def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    image_path = project_root / "dataset" / "images" / "train" / "frame_000763.jpg"
    label_path = project_root / "dataset" / "labels" / "train" / "frame_000763.txt"
    output_path = project_root / "predictions" / "ground_truth_frame_000763.jpg"

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
            if not 0 <= class_id < len(CLASS_NAMES):
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
            color=class_color(CLASS_NAMES[class_id]),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        name = CLASS_NAMES[class_id]
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
