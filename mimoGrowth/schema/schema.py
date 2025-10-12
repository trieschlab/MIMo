"""
Schemas for MIMo body composition.

These schemas are declarative, relational descriptions that map infant
measurements to every geom and body. Each size or position is defined in terms
of measurements, operations, or references. That makes the schema a single,
traceable source of truth: you can always follow a chain of references
back to the underlying measurements.

Notes:
- The ordering of size entries matches MuJoCo conventions.
- Very small constants are kept explicit.
- Only the 'left' side attributes are stored in the schema; the 'right'
    counterparts are generated via code at runtime to avoid duplication.
"""

from mimoGrowth.schema.ratios import RATIOS, RATIOS_V2


# Helper functions to build expression dictionaries. Each returns a dict
# encoding an operation ('$op') or a reference ('$ref') with its arguments.
def neg(*args): return {"$op": "neg", "args": list(args)}
def add(*args): return {"$op": "add", "args": list(args)}
def sub(*args): return {"$op": "sub", "args": list(args)}
def mul(*args): return {"$op": "mul", "args": list(args)}
def div(*args): return {"$op": "div", "args": list(args)}
def mean(*args): return {"$op": "mean", "args": list(args)}
def ref(*path): return {"$ref": list(path)}


# Since all torso geoms are based on the hip breadth infant measurement with
# ratios from the original model, this function makes it a bit easier to
# create the size definitions.
def torso_geom_size(geom: str) -> list:
    return [
        mul("hip_breadth", RATIOS[f"{geom}_scale"], RATIOS[f"{geom}_prop"]),
        mul("hip_breadth", RATIOS[f"{geom}_scale"], 1 - RATIOS[f"{geom}_prop"])
    ]


# The z-position of the torso geoms seem to be mostly manual tweaks. They are
# stored in the ratios and this function allows to remove redundant code.
def torso_geom_pos_z(geom):
    return mul(ref("geoms", geom, "size", 0), RATIOS[f"{geom}_pos"])


# There are no infant measurements for the hand and foot height. Therefore, we
# compute it based on the mean of length and breadth.
HAND_HEIGHT = mul(mean("hand_length", "hand_breadth"), RATIOS["hand_height"])
FOOT_HEIGHT = mul(mean("foot_length", "foot_breadth"), RATIOS["foot_height"])
HAND_HEIGHT_V2 = mul(
    mean("hand_length", "hand_breadth"),
    RATIOS_V2["hand_height"]
)

# This factor describes how much the phalanxes should shrink towards the tip.
PHA_SHRINK = 0.95

# These factors describes the difference in finger length relative to the
# middle finger. Keep in mind that the index and ring finger are identical.
LEN_MID_IDX = 0.9
LEN_MID_LITTLE = 0.88

# These factors describes the difference in finger diameter relative to the
# middle finger. Keep in mind that the index and ring finger are identical.
DIAM_MID_IDX = 0.95
DIAM_MID_LITTLE = 0.91

# This ratio describes the proportions of palm and (middle) finger length
# relative to the total hand length.
PALM_RATIO = 0.5
PALM_LEN = mul("hand_length", PALM_RATIO)
FINGER_LEN = mul("hand_length", 1 - PALM_RATIO)

# Define a small constant to subtract from some geom vectors so that the
# individual parts won't have a visual overlap. This value is from the
# original MIMo model.
EPSILON = 0.0001

SCHEMA_GEOMS = {

    # === TORSO ===

    "lb": {
        "size": torso_geom_size("lb"),
        "pos": [-0.002, 0, torso_geom_pos_z("lb")],
    },
    "cb": {
        "size": torso_geom_size("cb"),
        "pos": [0.005, 0, torso_geom_pos_z("cb")]
    },
    "ub1": {
        "size": torso_geom_size("ub1"),
        "pos": [0.007, 0, torso_geom_pos_z("ub1")]
    },
    "ub2": {
        "size": torso_geom_size("ub2"),
        "pos": [0.004, 0, torso_geom_pos_z("ub2")]
    },
    "ub3": {
        "size": torso_geom_size("ub3"),
        "pos": [0, 0, 0]
    },

    # === HEAD & EYES ===

    "head": {
        "size": [mul("head_circumference", RATIOS["head"])],
        "pos": [
            mul(ref("geoms", "head", "size", 0), RATIOS["head_pos"]),
            0,
            ref("geoms", "head", "size", 0)
        ]
    },
    "geom:left_eye1": {
        "size": [mul(ref("geoms", "head", "size", 0), RATIOS["eye"])],
        "pos": [0, 0, 0]
    },

    # === UPPER / LOWER ARM ===

    "left_uarm1": {
        "size": [
            mul("upper_arm_circumference", RATIOS["uarm_rad"]),
            mul(
                sub("shoulder_elbow_length", "upper_arm_circumference"),
                RATIOS["uarm_len"]
            )
        ],
        "pos": [0, 0, ref("geoms", "left_uarm1", "size", 1)]
    },
    "left_larm": {
        "size": [
            mul("forearm_circumference", RATIOS["larm_rad"]),
            mul(
                sub(
                    "elbow_hand_length",
                    "hand_length",
                    "forearm_circumference"
                ),
                RATIOS["larm_len"]
            )
        ],
        "pos": [0, 0, ref("geoms", "left_larm", "size", 1)]
    },

    # === HAND ===

    "geom:left_hand1": {
        "size": [
            mul("maximum_fist_breadth", RATIOS["hand1_breadth"]),
            HAND_HEIGHT,
            mul("hand_length", RATIOS["hand1_length"])
        ],
        "pos": [
            div(HAND_HEIGHT, 2),
            0,
            ref("geoms", "geom:left_hand1", "size", 2)
        ]
    },
    "geom:left_hand2": {
        "size": [
            add(HAND_HEIGHT, EPSILON * 2),
            mul("maximum_fist_breadth", RATIOS["hand1_breadth"])
        ],
        "pos": [
            div(HAND_HEIGHT, 2),
            0,
            mul(ref("geoms", "geom:left_hand1", "size", 2), 2)
        ]
    },
    "geom:left_fingers1": {
        "size": [
            sub("hand_breadth", 2 * EPSILON),
            HAND_HEIGHT,
            mul("hand_length", RATIOS["fingers1_length"])
        ],
        "pos": [0, 0, ref("geoms", "geom:left_fingers1", "size", 2)]
    },
    "geom:left_fingers2": {
        "size": [add(HAND_HEIGHT, EPSILON * 2), "hand_breadth"],
        "pos": [0, 0, mul(ref("geoms", "geom:left_fingers1", "size", 2), 2)]
    },

    # === UPPER / LOWER LEG ===

    "geom:left_upper_leg1": {
        "size": [
            mul("mid_thigh_circumference", RATIOS["uleg_rad"]),
            mul(
                sub("rump_knee_length", "mid_thigh_circumference"),
                RATIOS["uleg_len"]
            )
        ],
        "pos": [
            0,
            0,
            mul(
                ref("geoms", "geom:left_upper_leg1", "size", 1),
                RATIOS["uleg_pos"]
            )
        ]
    },
    "geom:left_lower_leg1": {
        "size": [
            "calf_circumference",
            mul("knee_sole_length", RATIOS["lleg1_len"])
        ],
        "pos": [
            0,
            0,
            neg(add(ref("geoms", "geom:left_lower_leg1", "size", 1)))
        ]
    },
    "geom:left_lower_leg2": {
        "size": [
            "ankle_circumference",
            mul("knee_sole_length", RATIOS["lleg2_len"])
        ],
        "pos": [
            0,
            0,
            neg(sub(mul(ref("geoms", "geom:left_lower_leg1", "size", 1), 3)))
        ]
    },

    # === FOOT ===

    "geom:left_foot1": {
        "size": [sub("foot_breadth", EPSILON), sub(FOOT_HEIGHT, EPSILON)],
        "pos": [
            neg(mul(
                ref("geoms", "geom:left_foot2", "size", 0),
                RATIOS["foot_pos"]
            )),
            0,
            0
        ],
    },
    "geom:left_foot2": {
        "size": [
            mul("foot_length", RATIOS["foot_len"]),
            "foot_breadth",
            FOOT_HEIGHT
        ],
        "pos": [
            mul(
                ref("geoms", "geom:left_foot2", "size", 0),
                1 - RATIOS["foot_pos"]
            ),
            0,
            0
        ],
    },
    "geom:left_foot3": {
        "size": [sub(FOOT_HEIGHT, EPSILON), sub("foot_breadth", 2 * EPSILON)],
        "pos": [
            add(
                ref("geoms", "geom:left_foot2", "size", 0),
                ref("geoms", "geom:left_foot2", "pos", 0),
            ),
            0,
            0
        ],
    },
    "geom:left_toes1": {
        "size": [
            mul("foot_length", RATIOS["toes_len"]),
            sub("foot_breadth", EPSILON),
            sub(FOOT_HEIGHT, EPSILON)
        ],
        "pos": [ref("geoms", "geom:left_toes1", "size", 0), 0, 0]
    },
    "geom:left_toes2": {
        "size": [FOOT_HEIGHT, "foot_breadth"],
        "pos": [mul(ref("geoms", "geom:left_toes1", "size", 0), 2), 0, 0]
    },

}

SCHEMA_BODIES = {

    # === TORSO ===

    "hip": {
        "pos": [0, 0, 0]
    },
    "lower_body": {
        "pos": [
            0.002,
            0,
            mul(add(
                ref("geoms", "lb", "size", 0),
                ref("geoms", "cb", "size", 0)
            ), RATIOS["lower_body"])
        ]
    },
    "upper_body": {
        "pos": [
            -0.002,
            0,
            mul(add(
                ref("geoms", "cb", "size", 0),
                ref("geoms", "ub1", "size", 0)
            ), RATIOS["upper_body"])
        ]
    },
    "chest": {
        "pos": [0, 0, torso_geom_pos_z("ub3")]
    },

    # === HEAD & EYES ===

    "head": {
        "pos": [
            0,
            0,
            add(
                ref("geoms", "ub3", "size", 0),
                ref("bodies", "chest", "pos", 2),
            )
        ]
    },
    "left_eye": {
        "pos": [
            mul(ref("geoms", "head", "size", 0), RATIOS["eye_pos"][i])
            for i in range(3)
        ]
    },

    # === UPPER / LOWER ARM ===

    "left_upper_arm": {
        "pos": [
            0,
            add(
                ref("geoms", "ub3", "size", 0),
                ref("geoms", "ub3", "size", 1),
                # The multiplication determines the overlap between shoulder
                # and upper body. Multiplying by 1 means no overlap.
                mul(ref("geoms", "left_uarm1", "size", 0), 0.25)
            ),
            0
        ]
    },
    "left_lower_arm": {
        "pos": [0, 0, mul(ref("geoms", "left_uarm1", "size", 1), 2)]
    },

    # === HAND ===

    "left_hand": {
        "pos": [
            0,
            0,  # 0.007,
            mul(add(
                ref("geoms", "left_larm", "size", 0),
                mul(ref("geoms", "left_larm", "size", 1), 2),
            ), RATIOS["hand_pos"])
        ]
    },
    "left_fingers": {
        "pos": [0, 0, ref("geoms", "geom:left_hand2", "pos", 2)]
    },

    # === UPPER / LOWER LEG ===

    "left_upper_leg": {
        "pos": [
            0.005,
            mul(ref("geoms", "lb", "size", 1), RATIOS["uleg_shift"]),
            -0.007
        ]
    },
    "left_lower_leg": {
        "pos": [
            0,
            0,
            neg(add(
                mul(ref("geoms", "geom:left_upper_leg1", "size", 0), 2),
                ref("geoms", "geom:left_upper_leg1", "size", 1)
            ))
        ]
    },

    # === FOOT ===

    "left_foot": {
        "pos": [
            0,
            0,
            add(
                ref("geoms", "geom:left_lower_leg1", "pos", 2),
                ref("geoms", "geom:left_lower_leg2", "pos", 2),
            )
        ]
    },
    "left_toes": {
        "pos": [ref("geoms", "geom:left_foot3", "pos", 0), 0, 0]
    }

}

SCHEMA_JOINTS = {

    # === HIP ===
    "robot:hip_lean1": {
        "pos": [0, 0, neg(ref("geoms", "cb", "size", 0))]
    },
    "robot:hip_rot1": {
        "pos": [0, 0, neg(ref("geoms", "cb", "size", 0))]
    },
    "robot:hip_bend1": {
        "pos": [0, 0, neg(ref("geoms", "cb", "size", 0))]
    },
    "robot:hip_lean2": {
        "pos": [0, 0, neg(ref("geoms", "ub1", "size", 0))]
    },
    "robot:hip_rot2": {
        "pos": [0, 0, neg(ref("geoms", "ub1", "size", 0))]
    },
    "robot:hip_bend2": {
        "pos": [0, 0, neg(ref("geoms", "ub1", "size", 0))]
    },

    # === HEAD ===
    "robot:head_tilt": {
        "pos": [0, 0, mul(ref("bodies", "left_eye", "pos", 1), 0.5)]
    },
    "robot:head_tilt_side": {
        "pos": [0, 0, ref("bodies", "left_eye", "pos", 1)]
    },

    # == SHOULDER ===
    "robot:left_shoulder_horizontal": {
        "pos": [0, 0, 0]
    },

    # === HAND ===
    "robot:left_hand1": {
        "pos": [0, ref("bodies", "left_hand", "pos", 1), 0]
    },

    # === FOOT ===
    "robot:left_foot1": {
        "pos": [0, 0, mul(FOOT_HEIGHT, 1.5)]
    },
    "robot:left_foot2": {
        "pos": [0, 0, mul(FOOT_HEIGHT, 1.5)]
    },
    "robot:left_foot3": {
        "pos": [0, 0, mul(FOOT_HEIGHT, 1.5)]
    },

}

SCHEMA_SITES = {

    # === VESTIBULAR ===
    "vestibular": {
        "pos": [0.01, 0, ref("geoms", "head", "size", 0)]
    },

    # === BODY_25 ===
    "BODY_25:Nose": {
        "pos": [
            mul(ref("geoms", "head", "size", 0), RATIOS["BODY_25:Nose_x"]),
            0.,
            mul(ref("geoms", "head", "size", 0), RATIOS["BODY_25:Nose_z"])
        ]
    },
    "BODY_25:Neck": {
        "pos": [0., 0., 0.]
    },
    "BODY_25:RShoulder": {
        "pos": [0., 0., 0.]
    },
    "BODY_25:RElbow": {
        "pos": [0., 0., 0.]
    },
    "BODY_25:RWrist": {
        "pos": [0., 0., 0.]
    },
    "BODY_25:LShoulder": {
        "pos": [0., 0., 0.]
    },
    "BODY_25:LElbow": {
        "pos": [0., 0., 0.]
    },
    "BODY_25:LWrist": {
        "pos": [0., 0., 0.]
    },
    "BODY_25:MidHip": {
        "pos": [0.005, 0, -0.007]
    },
    "BODY_25:RHip": {
        "pos": [0., 0., 0.]
    },
    "BODY_25:RKnee": {
        "pos": [0., 0., 0.]
    },
    "BODY_25:RAnkle": {
        "pos": [
            ref("joints", "robot:left_foot3", "pos", 0),
            ref("joints", "robot:left_foot3", "pos", 1),
            ref("joints", "robot:left_foot3", "pos", 2)
        ]  
    },
    "BODY_25:LHip": {
        "pos": [0., 0., 0.]
    },
    "BODY_25:LKnee": {
        "pos": [0., 0., 0.]
    },
    "BODY_25:LAnkle": {
        "pos": [
            ref("joints", "robot:left_foot3", "pos", 0),
            ref("joints", "robot:left_foot3", "pos", 1),
            ref("joints", "robot:left_foot3", "pos", 2)
        ]  
    },
    "BODY_25:REye": {
        "pos": [0., 0., 0.]
    },
    "BODY_25:LEye": {
        "pos": [0., 0., 0.]
    },
    "BODY_25:REar": {
        "pos": [
            0.01,
            neg(mul(ref("geoms", "head", "size", 0), RATIOS["BODY_25:Ear_y"])),
            mul(ref("geoms", "head", "size", 0), RATIOS["BODY_25:Ear_z"])
        ]
    },
    "BODY_25:LEar": {
        "pos": [
            0.01,
            mul(ref("geoms", "head", "size", 0), RATIOS["BODY_25:Ear_y"]),
            mul(ref("geoms", "head", "size", 0), RATIOS["BODY_25:Ear_z"])
        ]
    },
    "BODY_25:LBigToe": {
        "pos": [
            mul(ref("geoms", "geom:left_foot2", "size", 0), RATIOS["BODY_25:BigToe_x"]),
            neg(mul(ref("geoms", "geom:left_foot2", "size", 1), RATIOS["BODY_25:Toes_y"])),
            0
        ]
    },
    "BODY_25:LSmallToe": {
        "pos": [
            mul(ref("geoms", "geom:left_foot2", "size", 0), RATIOS["BODY_25:SmallToe_x"]),
            mul(ref("geoms", "geom:left_foot2", "size", 1), RATIOS["BODY_25:Toes_y"]),
            0
        ]
    },
    "BODY_25:LHeel": {
        "pos": [neg(mul(ref("geoms", "geom:left_foot2", "size", 0)), RATIOS["BODY_25:Heel_x"]), 0,0]
    },
    "BODY_25:RBigToe": {
        "pos": [
            mul(ref("geoms", "geom:left_foot2", "size", 0), RATIOS["BODY_25:BigToe_x"]),
            mul(ref("geoms", "geom:left_foot2", "size", 1), RATIOS["BODY_25:Toes_y"]),
            0
        ]
    },
    "BODY_25:RSmallToe": {
        "pos": [
            mul(ref("geoms", "geom:left_foot2", "size", 0), RATIOS["BODY_25:SmallToe_x"]),
            neg(mul(ref("geoms", "geom:left_foot2", "size", 1), RATIOS["BODY_25:Toes_y"])),
            0
        ]
    },
    "BODY_25:RHeel": {
        "pos": [neg(mul(ref("geoms", "geom:left_foot2", "size", 0)), RATIOS["BODY_25:Heel_x"]), 0,0]
    },
}


SCHEMA_GEOMS_V2 = {

    # === HAND ===

    "geom:left_hand1": {
        "size": [
            mul("hand_breadth", RATIOS_V2["hand_breadth"]),
            HAND_HEIGHT_V2,
            sub(PALM_LEN, HAND_HEIGHT_V2)
        ],
        "pos": [0, 0, 0],
    },
    "geom:left_hand2": {
        "size": [
            mul("hand_breadth", 1 - RATIOS_V2["hand_breadth"]),
            HAND_HEIGHT_V2,
            mul(ref("geoms", "geom:left_hand1", "size", 2), 0.5)
        ],
        "pos": [
            neg("hand_breadth"),
            0,
            neg(sub(
                ref("geoms", "geom:left_hand1", "size", 2),
                ref("geoms", "geom:left_hand2", "size", 2)
            ))
        ]
    },
    "geom:left_hand3": {
        "size": [
            HAND_HEIGHT_V2,
            sub(ref("geoms", "geom:left_hand1", "size", 0), EPSILON)
        ],
        "pos": [0, 0, ref("geoms", "geom:left_hand1", "size", 2)]
    },
    "geom:left_hand4": {
        "size": [
            HAND_HEIGHT_V2,
            sub("hand_breadth", EPSILON)
        ],
        "pos": [
            neg(ref("geoms", "geom:left_hand2", "size", 0)),
            0,
            neg(ref("geoms", "geom:left_hand1", "size", 2))
        ]
    },

    # === FIRST / INDEX FINGER

    "geom:left_ffknuckle1": {
        "size": [
            mul("middle_finger_diameter", DIAM_MID_IDX),
            mul(FINGER_LEN, 0.5, LEN_MID_IDX)
        ],
        "pos": [0, 0, ref("geoms", "geom:left_ffknuckle1", "size", 1)],
    },
    "geom:left_ffmiddle1": {
        "size": [
            mul(ref("geoms", "geom:left_ffknuckle1", "size", 0), PHA_SHRINK),
            mul(ref("geoms", "geom:left_ffknuckle1", "size", 1), 0.5),
        ],
        "pos": [0, 0, ref("geoms", "geom:left_ffmiddle1", "size", 1)],
    },
    "geom:left_ffdistal1": {
        "size": [
            mul(ref("geoms", "geom:left_ffmiddle1", "size", 0), PHA_SHRINK),
            sub(
                mul(ref("geoms", "geom:left_ffknuckle1", "size", 1), 0.5),
                mul(ref("geoms", "geom:left_ffdistal1", "size", 0), 0.5),
            ),
        ],
        "pos": [0, 0, ref("geoms", "geom:left_ffdistal1", "size", 1)],
    },

    # === MIDDLE FINGER ===

    "geom:left_mfknuckle1": {
        "size": [
            "middle_finger_diameter",
            mul(FINGER_LEN, 0.5)
        ],
        "pos": [
            0,
            0,
            ref("geoms", "geom:left_mfknuckle1", "size", 1)
        ],
    },
    "geom:left_mfmiddle1": {
        "size": [
            mul(ref("geoms", "geom:left_mfknuckle1", "size", 0), PHA_SHRINK),
            mul(ref("geoms", "geom:left_mfknuckle1", "size", 1), 0.5),
        ],
        "pos": [
            0,
            0,
            ref("geoms", "geom:left_mfmiddle1", "size", 1)
        ],
    },
    "geom:left_mfdistal1": {
        "size": [
            mul(ref("geoms", "geom:left_mfmiddle1", "size", 0), PHA_SHRINK),
            sub(
                mul(ref("geoms", "geom:left_mfknuckle1", "size", 1), 0.5),
                mul(ref("geoms", "geom:left_mfdistal1", "size", 0), 0.5),
            ),
        ],
        "pos": [
            0,
            0,
            ref("geoms", "geom:left_mfdistal1", "size", 1)
        ]
    },

    # === RING FINGER ===

    "geom:left_rfknuckle1": {
        "size": [
            mul("middle_finger_diameter", DIAM_MID_IDX),
            mul(FINGER_LEN, 0.5, LEN_MID_IDX)
        ],
        "pos": [0, 0, ref("geoms", "geom:left_rfknuckle1", "size", 1)],
    },
    "geom:left_rfmiddle1": {
        "size": [
            mul(ref("geoms", "geom:left_rfknuckle1", "size", 0), PHA_SHRINK),
            mul(ref("geoms", "geom:left_rfknuckle1", "size", 1), 0.5),
        ],
        "pos": [0, 0, ref("geoms", "geom:left_rfmiddle1", "size", 1)],
    },
    "geom:left_rfdistal1": {
        "size": [
            mul(ref("geoms", "geom:left_rfmiddle1", "size", 0), PHA_SHRINK),
            sub(
                mul(ref("geoms", "geom:left_rfknuckle1", "size", 1), 0.5),
                mul(ref("geoms", "geom:left_rfdistal1", "size", 0), 0.5),
            ),
        ],
        "pos": [0, 0, ref("geoms", "geom:left_rfdistal1", "size", 1)],
    },

    # === LITTLE FINGER ===

    "geom:left_lfmetacarpal1": {
        "size": [
            mul(
                ref("geoms", "geom:left_hand2", "size", 0),
                RATIOS_V2["little_finger"]
            ),
            add(HAND_HEIGHT_V2, EPSILON),
            ref("geoms", "geom:left_hand2", "size", 2),
        ],
        "pos": [
            neg(add(
                ref("geoms", "geom:left_hand1", "size", 0),
                mul(ref("geoms", "geom:left_hand2", "size", 0), 2),
                neg(ref("geoms", "geom:left_lfmetacarpal1", "size", 0)),
                ref("bodies", "left_lfmetacarpal", "pos", 0),
                EPSILON
            )),
            0,
            neg(ref("bodies", "left_lfmetacarpal", "pos", 2))
        ],
    },
    "geom:left_lfmetacarpal2": {
        "size": [
            add(HAND_HEIGHT_V2, EPSILON),
            sub(ref("geoms", "geom:left_lfmetacarpal1", "size", 0), EPSILON)
        ],
        "pos": [
            ref("geoms", "geom:left_lfmetacarpal1", "pos", 0),
            0,
            add(
                ref("geoms", "geom:left_lfmetacarpal1", "size", 2),
                ref("geoms", "geom:left_lfmetacarpal1", "pos", 2)
            )
        ],
    },
    "geom:left_lfknuckle1": {
        "size": [
            mul("middle_finger_diameter", DIAM_MID_LITTLE),
            mul(FINGER_LEN, 0.5, LEN_MID_LITTLE)
        ],
        "pos": [0, 0, ref("geoms", "geom:left_lfknuckle1", "size", 1)],
    },
    "geom:left_lfmiddle1": {
        "size": [
            mul(ref("geoms", "geom:left_lfknuckle1", "size", 0), PHA_SHRINK),
            mul(ref("geoms", "geom:left_lfknuckle1", "size", 1), 0.5),
        ],
        "pos": [0, 0, ref("geoms", "geom:left_lfmiddle1", "size", 1)],
    },
    "geom:left_lfdistal1": {
        "size": [
            mul(ref("geoms", "geom:left_lfmiddle1", "size", 0), PHA_SHRINK),
            sub(
                mul(ref("geoms", "geom:left_lfknuckle1", "size", 1), 0.5),
                mul(ref("geoms", "geom:left_lfdistal1", "size", 0), 0.5),
            ),
        ],
        "pos": [0, 0, ref("geoms", "geom:left_lfdistal1", "size", 1)],
    },

    # === THUMB ===

    "geom:left_thbase1": {
        "size": [
            "thumb_diameter",
            mul(ref("geoms", "geom:left_mfknuckle1", "size", 1), 0.75)  # here
        ],
        "pos": [0, 0, ref("geoms", "geom:left_thbase1", "size", 1)],
    },
    "geom:left_thhub1": {
        "size": [
            mul(ref("geoms", "geom:left_thbase1", "size", 0), PHA_SHRINK),
            mul(ref("geoms", "geom:left_thbase1", "size", 1), 0.7)  # herer
        ],
        "pos": [0, 0, ref("geoms", "geom:left_thhub1", "size", 1)]
    },
    "geom:left_thdistal1": {
        "size": [
            mul(ref("geoms", "geom:left_thhub1", "size", 0), PHA_SHRINK),
            mul(ref("geoms", "geom:left_thbase1", "size", 1), 0.65)  # here
        ],
        "pos": [0, 0, ref("geoms", "geom:left_thdistal1", "size", 1)],
    },

    # === TOES ===

    "geom:left_toes1": {
        "size": [
            mul("foot_length", RATIOS_V2["toes_len"]),
            mul("foot_breadth", RATIOS_V2["toes_breadth"]),
            sub(FOOT_HEIGHT, EPSILON)
        ],
        "pos": [ref("geoms", "geom:left_toes1", "size", 0)]
    },
    "geom:left_toes2": {
        "size": [
            FOOT_HEIGHT,
            mul(
                ref("geoms", "geom:left_toes1", "size", 1),
                RATIOS_V2["toes_breadth_diff"]
            )
        ],
        "pos": [mul(ref("geoms", "geom:left_toes1", "size", 0), 2)]
    },
    "geom:left_big_toe1": {
        "size": [
            mul("foot_length", RATIOS_V2["toes_len"]),
            mul("foot_breadth", RATIOS_V2["big_toe_breadth"]),
            sub(FOOT_HEIGHT, EPSILON)
        ],
        "pos": [ref("geoms", "geom:left_big_toe1", "size", 0)]
    },
    "geom:left_big_toe2": {
        "size": [
            FOOT_HEIGHT,
            mul(
                ref("geoms", "geom:left_big_toe1", "size", 1),
                RATIOS_V2["big_toe_breadth_diff"]
            )
        ],
        "pos": [mul(ref("geoms", "geom:left_big_toe1", "size", 0), 2)]
    },

}

SCHEMA_BODIES_V2 = {

    # === HAND ===

    "left_hand": {
        "pos": [
            mul("hand_breadth", RATIOS_V2["hand_pos_x"]),
            0,  # mul(HAND_HEIGHT_V2, RATIOS_V2["hand_pos_y"]),
            mul(add(
                mul(ref("geoms", "left_larm", "size", 0), 2),
                mul(ref("geoms", "left_larm", "size", 1), 2),
            ), RATIOS_V2["hand_pos_z"])
        ]
    },

    # === FIRST / INDEX FINGER ===

    "left_ffknuckle": {
        "pos": [
            sub(
                ref("geoms", "geom:left_hand1", "size", 0),
                ref("geoms", "geom:left_ffknuckle1", "size", 0),
                EPSILON
            ),
            0,
            ref("geoms", "geom:left_hand1", "size", 2),
        ]
    },
    "left_ffmiddle": {
        "pos": [
            0,
            0,
            mul(ref("geoms", "geom:left_ffknuckle1", "size", 1), 2)
        ]
    },
    "left_ffdistal": {
        "pos": [
            0,
            0,
            mul(ref("geoms", "geom:left_ffmiddle1", "size", 1), 2)
        ]
    },

    # === MIDDLE FINGER ===

    "left_mfknuckle": {
        "pos": [
            0,
            0,
            ref("geoms", "geom:left_hand1", "size", 2),
        ]
    },
    "left_mfmiddle": {
        "pos": [
            0,
            0,
            mul(ref("geoms", "geom:left_mfknuckle1", "size", 1), 2)
        ]
    },
    "left_mfdistal": {
        "pos": [
            0,
            0,
            mul(ref("geoms", "geom:left_mfmiddle1", "size", 1), 2)
        ]
    },

    # === RING FINGER ===

    "left_rfknuckle": {
        "pos": [
            neg(sub(
                ref("geoms", "geom:left_hand1", "size", 0),
                ref("geoms", "geom:left_rfknuckle1", "size", 0),
                EPSILON
            )),
            0,
            ref("geoms", "geom:left_hand1", "size", 2),
        ]
    },
    "left_rfmiddle": {
        "pos": [0, 0, mul(ref("geoms", "geom:left_rfknuckle1", "size", 1), 2)]
    },
    "left_rfdistal": {
        "pos": [0, 0, mul(ref("geoms", "geom:left_rfmiddle1", "size", 1), 2)]
    },

    # === LITTLE FINGER ===

    "left_lfmetacarpal": {
        "pos": [
            mul("hand_breadth", RATIOS_V2["lf_body1"]),
            0,
            mul("hand_breadth", RATIOS_V2["lf_body2"]),
        ]
    },
    "left_lfknuckle": {
        "pos": [
            neg(add(
                ref("bodies", "left_lfmetacarpal", "pos", 0),
                ref("geoms", "geom:left_hand1", "size", 0),
                mul(
                    ref("geoms", "geom:left_hand2", "size", 0),
                    1 / RATIOS_V2["little_finger"]
                ),
            )),
            0,
            add(
                neg(ref("bodies", "left_lfmetacarpal", "pos", 2)),
                mul(ref("geoms", "geom:left_lfmetacarpal1", "size", 2), 1)
            )
        ]
    },
    "left_lfmiddle": {
        "pos": [0, 0, mul(ref("geoms", "geom:left_lfknuckle1", "size", 1), 2)]
    },
    "left_lfdistal": {
        "pos": [0, 0, mul(ref("geoms", "geom:left_lfmiddle1", "size", 1), 2)]
    },

    # === THUMB ===

    "left_thbase": {
        "pos": [
            mul("hand_breadth", RATIOS_V2["thumb_body1"]),
            mul(HAND_HEIGHT_V2, RATIOS_V2["thumb_body2"]),
            mul("hand_length", RATIOS_V2["thumb_body3"]),
        ]
    },
    "left_thhub": {
        "pos": [0, 0, mul(ref("geoms", "geom:left_thbase1", "size", 1), 2)]
    },
    "left_thdistal": {
        "pos": [0, 0, mul(ref("geoms", "geom:left_thhub1", "size", 1), 2)]
    },

    # === TOES ===

    "left_toes": {
        "pos": [
            ref("geoms", "geom:left_foot3", "pos", 0),
            mul("foot_breadth", 0.3)
        ],
    },
    "left_big_toe": {
        "pos": [
            ref("geoms", "geom:left_foot3", "pos", 0),
            neg(mul("foot_breadth", 0.7))
        ]
    }

}

SCHEMA_JOINTS_V2 = {

    # === HAND ===
    "robot:left_hand1": {
        "pos": [
            neg(ref("bodies", "left_hand", "pos", 0)),
            0,  # neg(ref("bodies", "left_hand", "pos", 0)),
            mul(neg(ref("geoms", "geom:left_hand1", "size", 2)), 3/5)
        ]
    },
    "robot:left_hand2": {
        "pos": [
            neg(ref("bodies", "left_hand", "pos", 0)),
            0,  # missing
            mul(neg(ref("geoms", "geom:left_hand1", "size", 2)), 4/5)
        ]
    },
    "robot:left_hand3": {  # done
        "pos": [
            neg(ref("bodies", "left_hand", "pos", 0)),
            0,
            neg(ref("geoms", "geom:left_hand1", "size", 2))  # temp/wrong
        ]
    }

}

SCHEMA_SITES_V2 = {
    "BODY_25:RWrist": {
        "pos": [
            ref("joints","robot:left_hand3","pos", 0),
            ref("joints","robot:left_hand3","pos", 1),
            ref("joints","robot:left_hand3","pos", 2),
        ]   
    },
    "BODY_25:LWrist": {
        "pos": [
            ref("joints","robot:left_hand3","pos", 0),
            ref("joints","robot:left_hand3","pos", 1),
            ref("joints","robot:left_hand3","pos", 2),
        ]   
    },
    "BODY_25:LBigToe": {
        "pos": [0, 0, 0]
    },
    "BODY_25:LSmallToe": {
        "pos": [0, mul(ref("geoms", "geom:left_toes1", "size", 1), 0.45), 0]
    },
    "BODY_25:RBigToe": {
        "pos": [0, 0, 0]
    },
    "BODY_25:RSmallToe": {
        "pos": [0, neg(mul(ref("geoms", "geom:left_toes1", "size", 1), 0.45)), 0]
    },
}

SCHEMA = {
    "geoms": SCHEMA_GEOMS,
    "bodies": SCHEMA_BODIES,
    "joints": SCHEMA_JOINTS,
    "sites": SCHEMA_SITES
}

# Merge the schema dicts for MIMo_v2. Note that if a key appears
# twice, the key from the right-hand dict will be used.
SCHEMA_V2 = {
    "geoms": SCHEMA_GEOMS | SCHEMA_GEOMS_V2,
    "bodies": SCHEMA_BODIES | SCHEMA_BODIES_V2,
    "joints": SCHEMA_JOINTS | SCHEMA_JOINTS_V2,
    "sites": SCHEMA_SITES | SCHEMA_SITES_V2,
}

# Remove entries from schema_v2 that are not overwritten and
# are also not defined in the second version of MIMo.
SCHEMA_V2["geoms"].pop("geom:left_fingers1")
SCHEMA_V2["geoms"].pop("geom:left_fingers2")
SCHEMA_V2["bodies"].pop("left_fingers")
