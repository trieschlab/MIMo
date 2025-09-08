"""
Stores various ratios that are used to simulate the growth of MIMo.

Ratios are used for several reasons:
- To maintain the manual tweaks from the original model by capturing
    differences between measurements and sizes from the original MIMo.
- To estimate sizes where direct measurements are not available.
- To define relationships between geom sizes or positions.

Each ratio is documented in detail to explain its purpose and usage.

Note that it is always necessary to convert measurements since MuJoCo uses
a different format to represent sizes.
"""

from mimoGrowth.utils import mj_unit
import numpy as np

# Store converted measurements that are used multiples times.
MJ_HIP_BREADTH = mj_unit(17.1, "cm", "len")
MJ_UARM_CIRC = mj_unit(14.7, "cm", "circ")
MJ_FOREARM_CIRC = mj_unit(14.5, "cm", "circ")
MJ_HAND_LENGTH = mj_unit(9.3, "cm", "len")
MJ_MAX_FIST_BREADTH = mj_unit(5.5, "cm", "len")
MJ_HAND_BREADTH = mj_unit(4.6, "cm", "len")
MJ_MID_THIGH_CIRC = mj_unit(24.4, "cm", "circ")
MJ_KNEE_SOLE_LEN = mj_unit(21.6, "cm", "len")
MJ_FOOT_LEN = mj_unit(11.9, "cm", "len")
MJ_FOOT_BREADTH = mj_unit(5, "cm", "len")

RATIOS = {

    # These ratios describe the differences between the hip breadth measurement
    # and the breadth of the torso geoms from the original MIMo model.
    "lb_scale": (0.048 + 0.043) / MJ_HIP_BREADTH,
    "cb_scale": (0.053 + 0.035) / MJ_HIP_BREADTH,
    "ub1_scale": (0.052 + 0.035) / MJ_HIP_BREADTH,
    "ub2_scale": (0.048 + 0.039) / MJ_HIP_BREADTH,
    "ub3_scale": (0.041 + 0.047) / MJ_HIP_BREADTH,

    # These ratios describe the proportions between the radius and half-length
    # of the torso geoms from the original MIMo model. We need these ratios to
    # figure out the radius of the geom torsos since we don't have
    # measurements for the hip depth.
    "lb_prop": 0.048 / (0.048 + 0.043),
    "cb_prop": 0.053 / (0.053 + 0.035),
    "ub1_prop": 0.052 / (0.052 + 0.035),
    "ub2_prop": 0.048 / (0.048 + 0.039),
    "ub3_prop": 0.041 / (0.041 + 0.047),

    # These ratios describes the z-positions of the torso geoms relative to
    # their own radius. They are used to correctly update the height of the
    # torso geoms while MIMo is growing.
    "lb_pos": 0.005 / 0.048,
    "cb_pos": -0.008 / 0.053,
    "ub1_pos": -0.032 / 0.052,
    "ub2_pos": 0.03 / 0.048,
    "ub3_pos": 0.09 / 0.041,

    # This ratio describes the z-position of the lower/upper body relative to
    # the radius of other torso geoms.
    "lower_body": 0.076 / (0.053 + 0.048),
    "upper_body": 0.091 / (0.052 + 0.053),

    # This ratio describes difference between the head radius of the original
    # model and the infant measurement for the head circumference.
    "head": 0.0735 / mj_unit(46.8, "cm", "circ"),

    # This ratio describes the x-pos of the head relative to the size of the
    # head geom.
    "head_pos": 0.01 / 0.0735,

    # This ratio describes the radius of the eye geom relative to the radius
    # of the head geom.
    "eye": 0.01125 / 0.0735,

    # This ratio describes the position of the eye relative to the radius of
    # the head geom.
    "eye_pos": np.array([0.07, 0.0245, 0.067375]) / (0.0735),

    # These ratios describes the difference between the upper arm of the
    # original MIMo and the infant measurements of upper arm circumference and
    # shoulder-elbow length. Notice that we need to subtract the circumference
    # measurement from the length measurement since the circumference will
    # already be a part of the length since the arm geoms are capsules.
    "uarm_rad": 0.024 / MJ_UARM_CIRC,
    "uarm_len": 0.0536 / (mj_unit(15.4, "cm", "len") - MJ_UARM_CIRC),

    # These ratios describes the difference between the lower arm of the
    # original model and the infant measurements of forearm circumference and
    # elbow-hand length. Notice that we subtract the radius similar to the
    # upper arm and we need to subtract the hand length from the elbow-hand
    # length in order to get only the data for the lower arm.
    "larm_rad": 0.023 / MJ_FOREARM_CIRC,
    "larm_len": 0.037 / (mj_unit(20.7 - 9.3, "cm", "len") - MJ_FOREARM_CIRC),

    # These ratios describe the hand/foot height relative to the mean of length
    # and breadth measurements of hand/foot.
    "hand_height": 0.01 / np.mean(mj_unit([9.3, 4.6], "cm", "len")),
    "foot_height": 0.01 / np.mean(mj_unit([5, 11.9], "cm", "len")),

    # This ratio describes the z-position of the hand geom relative to the
    # calculated z-position based on the lower arm.
    "hand_pos": 0.087 / (0.023 + 0.037 * 2),

    # These ratios describe the length/breadth of different hand geoms relative
    # to the infant measurements.
    "hand1_length": 0.0208 / MJ_HAND_LENGTH,
    "hand1_breadth": 0.0281 / MJ_MAX_FIST_BREADTH,
    "hand2_breadth": 0.0278 / MJ_MAX_FIST_BREADTH,
    "fingers1_length": .0207 / MJ_HAND_LENGTH,

    # Similar to upper and lower arm, these ratios describes the differences
    # between the original model and infant measurements for the leg geoms.
    "uleg_rad": 0.037 / MJ_MID_THIGH_CIRC,
    "uleg_len": 0.0625 / (mj_unit(21.3, "cm", "len") - MJ_MID_THIGH_CIRC),

    # This ratio describes the z-pos of the upper leg geom relative to the
    # length of the geom.
    "uleg_pos": -0.0645 / 0.0625,

    # These ratios describe the length of the lower leg geoms relative to the
    # knee-sole length from the infant measurements.
    "lleg1_len": 0.044 / MJ_KNEE_SOLE_LEN,
    "lleg2_len": 0.028 / MJ_KNEE_SOLE_LEN,

    # This ratio describes the offset from the heel geom based on the length
    # of the middle foot geom.
    "foot_pos": 0.016 / (0.016 + 0.019),

    # These ratios describe the length of the foot/toes geoms relative to the
    # foot length measurement from infants.
    "foot_len": 0.035 / MJ_FOOT_LEN,
    "toes_len": 0.007 / MJ_FOOT_LEN,

}

RATIOS_V2 = {

    # === HAND ===

    # This ratio describes the position of the hand body relative to hand
    # measurements or other body positions.
    "hand_pos_x": 0.007 / MJ_HAND_BREADTH,
    "hand_pos_y": 0.009 / .00584,  # hand height
    "hand_pos_z": 0.11032 / (0.023 * 2 + 0.037 * 2),

    # The palm of MIMo is splitted into two geoms. This ratio describes the
    # proportion between the geoms and the actual infant measurement.
    "hand_breadth": 0.01712 / mj_unit(4.6, "cm", "len"),

    # Since there is no hand height infant measurement we use the value fom the
    # original model and the mean of length/breadth measurement to compute a
    # ratio that can be used at any age.
    "hand_height": .00584 / np.mean(mj_unit([9.3, 4.6], "cm", "len")),

    # This ratio describes the difference between the box geoms of the little
    # finger and essentially decides the size of the gap.
    "little_finger": .00508 / .00588,

    # The joint of the little finger has a custom angle/position. Therefore,
    # we keep this position by computing ratios relative to the hand breadth.
    # The hand breadth measurement is arbitrarily chosen.
    "lf_body1": -0.01498 / MJ_HAND_BREADTH,
    "lf_body2": 0.0031 / MJ_HAND_BREADTH,

    # These ratios describe the thumb body position relative to the
    # hand breadth.
    "thumb_body1": 0.0123 / MJ_HAND_BREADTH,
    "thumb_body2": 0.00423 / .00584,  # hand height
    "thumb_body3": -0.01602 / MJ_HAND_LENGTH,

    # === FOOT ===

    # These ratios describe how the breadth of toes and big toe relate to the
    # infant measurements of the foot breadth.
    "toes_breadth": .016 / MJ_FOOT_BREADTH,
    "big_toe_breadth": 0.007 / MJ_FOOT_BREADTH,

    # This ratio describes the length of the toes geom relative to the foot
    # length measurement of the infants.
    "toes_len": .0095 / MJ_FOOT_LEN,

    # This ratio describes the difference in breadth of the toes and big toe
    # geoms from the original model to avoid visual overlap.
    "toes_breadth_diff": 0.0165 / 0.016,
    "big_toe_breadth_diff": 0.0075 / 0.007

}
