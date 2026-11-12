DEFECT_CLASSES = {
    "swivel_clip_rust",
    "suspension_insulator_bend",
    "contact_wire_dropper_clamp_rust",
    "bird_nest",
    "stay_tube_rust",
    "counterweight_balance_mark_missing",
    "tie_rod_rust",
    "catenory_suspension_bracket_rust",
    "catenory_wire_rust",
    "insulator_broken",
    "counterweight_misalignement",
    "registration_tube_dropper_break",
    "insulator_muddy",
    "catenory_suspension_clamp_rust",
    "bracket_tube_rust",
}

GOOD_COMPONENT = (0, 200, 0)
DEFECT_COLOR = (0, 0, 255)


def class_color(class_name: str) -> tuple[int, int, int]:
    return DEFECT_COLOR if class_name in DEFECT_CLASSES else GOOD_COMPONENT
