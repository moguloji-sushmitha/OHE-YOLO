DEFECT_CLASSES = {
    "bracket_tube_rust",
}

GOOD_COMPONENT = (0, 200, 0)
DEFECT_COLOR = (0, 0, 255)


def class_color(class_name: str) -> tuple[int, int, int]:
    return DEFECT_COLOR if class_name in DEFECT_CLASSES else GOOD_COMPONENT
