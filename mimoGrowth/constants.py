""" This module stores all constant values. """

import numpy as np


# Store the mean values for the age groups on the website. All entries except
# the last one are from the infant measurements. The last entry is the mean age
# of the first row from the children measurements.
# This list will be used to approximate the growth functions.
MEAN_MEASUREMENT_AGES = [1, 3, 7, 10, 13.5, 17.5, 21.5, 33]

# Define which measurements we need for MIMo. The numbers indicate
# the ID's from the website for the infant and children measurements.
# Notice that a few times the measurements had different names, but described
# essentially the same thing.
MAPPING_MEASUREMENTS = {
    "ankle_circumference": (584, 405),
    "calf_circumference": (583, 397),
    "elbow_hand_length": (563, 237),
    "foot_breadth": (587, 421),
    "foot_length": (586, 417),
    "forearm_circumference": (564, 245),
    "hand_breadth": (567, 265),
    "hand_length": (566, 261),
    "head_circumference": (557, 145),
    "hip_breadth": (579, 361),  # Hip Breadth at Trochanter
    "knee_sole_length": (582, 121),  # Knee Height
    "maximum_fist_breadth": (569, 309),
    "mid_thigh_circumference": (580, 381),  # Upper Thigh Circumference
    "rump_knee_length": (577, 117),  # Buttock-Knee Length
    "shoulder_elbow_length": (561, 221),
    "upper_arm_circumference": (562, 229),
}

# Store ratios that describe the difference between measurements and the
# original MIMo model. These ratios will be used to maintain all the small
# tweaks that were made by hand along any age.
RATIOS_MIMO_GEOMS = {
    # circum model / circumference measurement
    "head": [(0.0735 * 200 * np.pi) / 46.8],
    "upper_arm": [
        # circumference model / circumference measurement
        (0.024 * 200 * np.pi) / 14.7,
        # middle-length model / middle-length measurement
        (0.0536 * 2) / ((15.4 - (2 * (14.7 / (2 * np.pi)))) / 100),
    ],
    "lower_arm": [
        # circumference model / circumference measurement
        (0.023 * 200 * np.pi) / 14.5,
        # middle-length model / middle-length measurement
        (0.037 * 2) / (((20.7 - 9.3) - (2 * (14.5 / (2 * np.pi)))) / 100),
    ],
    "hand": [
        # length model / length measurement
        ((0.0208 * 2 + 0.0207 * 2 + 0.0102) * 100) / 9.3,
        # hand breadth model / hand breadth measurement
        (0.0228 * 2 * 100) / 4.6,
        # fist breadth model / fist breadth measurement
        (0.0281 * 2 * 100) / 5.5,
    ],
    "torso": [
        # breadth model / breadth measurement
        ((0.048 * 2 + 0.043 * 2) * 100) / 17.1,  # lb
        ((0.053 * 2 + 0.035 * 2) * 100) / 17.1,  # cb
        ((0.052 * 2 + 0.035 * 2) * 100) / 17.1,  # ub1
        ((0.048 * 2 + 0.039 * 2) * 100) / 17.1,  # ub2
        ((0.041 * 2 + 0.047 * 2) * 100) / 17.1   # ub3
    ],
    "upper_leg": [
        # circumference model / circumference measurement
        (0.037 * 200 * np.pi) / 24.4,
        # middle-length model / middle-length measurement
        (0.0625 * 2) / ((21.3 - (2 * (24.4 / (2 * np.pi)))) / 100),
    ],
    "lower_leg": [
        # circumference calve model / circumference calve measurement
        (0.029 * 200 * np.pi) / 18.4,
        # circumference ankle model / circumference ankle measurement
        (0.021 * 200 * np.pi) / 13.3,
        # length model (with foot) / length measurement
        (((0.02 + 0.029 + 0.044 * 2 + 0.021 + 0.028 * 2) * 100)) / 21.6,
    ],
    "foot": [
        # length model / length measurement
        (0.0249 + 0.035 * 2 + 0.007 * 2 + 0.01) * 100 / 11.9,
        # breadth model / breadth measurement
        (0.025 * 2 * 100) / 5,
    ]
}

# Store ratios that describe the difference between the body positions from the
# original model and the computed position based on other body parts.
# These ratios will be used to maintain all the small tweaks that were made
# by hand along any age.
RATIOS_MIMO_BODIES = {
    "head": 0.135 / 0.131,  # model pos / calculated pos
    # eye: model pos / model head circumference
    "eye": np.array([0.07, 0.0245, 0.067375]) / (0.0735),
    "upper_arm": [
        0.105 / 0.112,  # model y-pos / calculated y-pos
        0.093 / 0.09  # model z-pos / calculated z-pos
    ],
    "lower_arm": 0.1076 / 0.1082,  # model z-pos / calculated z-pos
    "hand": 0.087 / 0.097,  # model z-pos / calculated z-pos
    "lower_body": 0.076 / 0.053,  # model z-pos / radius cb geom
    "upper_body": 0.091 / 0.052,  # model z-pos / radius ub1 geom
    "upper_leg": 0.051 / 0.054,  # model y-pos / calculated y-pos
    "lower_leg": 0.135 / 0.133,  # model z-pos / calculated z-pos
    "foot": 0.177 / 0.178,  # model z-pos / calculated z-pos
}

# Use ratios between different body parts from the original model to infer
# sizes for which there are no direct measurements on the website.
RATIOS_DERIVED = {
    "eye": 0.01125 / 0.0735,  # radius eye / radius head
    "hand": [
        # height model / geometric mean of half-length and half-breadth of hand
        0.01 / np.sqrt((0.0932 / 2) * 0.0228),
        # hand1-to-fingers1 length ratio
        0.0208 / (0.0208 + 0.0207),
    ],
    "torso_size": [
        # radius-to-length ratio
        0.048 / (0.048 + 0.043),  # lb
        0.053 / (0.053 + 0.035),  # cb
        0.052 / (0.052 + 0.035),  # ub1
        0.048 / (0.048 + 0.039),  # ub2
        0.041 / (0.041 + 0.047),  # ub3
    ],
    "torso_pos": [
        # z-pos / radius
        (0.005 / 0.048),  # lb
        (-0.008 / 0.053),  # cb
        (-0.032 / 0.052),  # ub1
        (0.03 / 0.048),  # ub2
        (0.09 / 0.041),  # ub3
    ],
    "upper_leg": 0.0645 / 0.0625,  # model pos / model size
    "lower_leg": [
        0.044 / (0.044 + 0.028),  # lower_leg_1-to-lower_leg_2 ratio
        0.134 / 0.137,  # model pos / calculated pos
    ],
    "foot": [
        # half-height model / geometric mean of half-width and half-breadth
        0.01 / np.sqrt((0.1189 / 2) * 0.025),
        # foot2-to-length ratio
        (0.035 * 2) / (0.0249 + 0.035 * 2 + 0.007 * 2 + 0.01),
        # toes1-to-length ratio
        (0.007 * 2) / (0.0249 + 0.035 * 2 + 0.007 * 2 + 0.01),
        # x-pos foot1 / x-size foot2
        0.016 / 0.035,
    ]
}

# Map the keywords for body parts to the actual geom
# names they are intended for.
MAPPING_GEOM = {
    "head": ["head", ("geom:left_eye1", "geom:right_eye1")],
    "upper_arm": [("left_uarm1", "right_uarm1")],
    "lower_arm": [("left_larm", "right_larm")],
    "hand": [
        ("geom:right_hand1", "geom:left_hand1"),
        ("geom:right_hand2", "geom:left_hand2"),
        ("geom:right_fingers1", "geom:left_fingers1"),
        ("geom:right_fingers2", "geom:left_fingers2"),
    ],
    "torso": ["lb", "cb", "ub1", "ub2", "ub3"],
    "upper_leg": [("geom:left_upper_leg1", "geom:right_upper_leg1")],
    "lower_leg": [
        ("geom:left_lower_leg1", "geom:right_lower_leg1"),
        ("geom:left_lower_leg2", "geom:right_lower_leg2"),
    ],
    "foot": [
        ("geom:left_foot1", "geom:right_foot1"),
        ("geom:left_foot2", "geom:right_foot2"),
        ("geom:left_foot3", "geom:right_foot3"),
        ("geom:left_toes1", "geom:right_toes1"),
        ("geom:left_toes2", "geom:right_toes2"),
    ]
}

# Map geoms to the corresponding motors. Note that below are only 'right'
# geoms and motors stored. Since MIMo is symmetrical, the 'left' ones
# will be done via code.
MAPPING_MOTOR = {
    "cb": ["act:hip_bend", "act:hip_twist", "act:hip_lean"],
    "head": ["act:head_swivel", "act:head_tilt", "act:head_tilt_side"],
    "geom:right_eye1": [
        "act:right_eye_horizontal",
        "act:right_eye_vertical",
        "act:right_eye_torsional"
    ],
    "right_uarm1": [
        "act:right_shoulder_horizontal",
        "act:right_shoulder_abduction",
        "act:right_shoulder_internal"
    ],
    "right_larm": ["act:right_elbow"],
    "geom:right_hand1": [
        "act:right_wrist_rotation",
        "act:right_wrist_flexion",
        "act:right_wrist_ulnar",
        "act:right_fingers"
    ],
    "geom:right_upper_leg1": [
        "act:right_hip_flex",
        "act:right_hip_abduction",
        "act:right_hip_rotation"
    ],
    "geom:right_lower_leg1": ["act:right_knee"],
    "geom:right_foot2": [
        "act:right_foot_flexion",
        "act:right_foot_inversion",
        "act:right_foot_rotation",
        "act:right_toes",
    ]
}
