"""
Functions for physics-related calculations that affect internal simulation
values rather than visual changes.

Since there are no detailed strength measurements for infants, we chose to
compute the gear value (strength) based on the volume of the closest geom.
Studies have shown that there is a correlation between muscle size and
muscle strength.

Similarly, there are not detailed mass measurements for single body parts of
infants. Therefore, the mass of a geom is calculated based on the volume and
density with the assumption that the density of body parts remains the same
over time. The density is taken from the original MIMo model.

Includes:
- `calc_motor_gear`: Calculates the gear value for all motors.
- `calc_geom_masses`: Calculates the mass for all geoms.
"""

from mimoGrowth.utils import calc_volume

# Map geoms to the corresponding motors. Note that below are only 'left'
# geoms and motors stored. Since MIMo is symmetrical, the 'right' ones
# will be done via code.
MAPPING_MOTOR = {
    "cb": ["act:hip_bend", "act:hip_twist", "act:hip_lean"],
    "ub3": ["act:chest_twist", "act:chest_lean"],
    "head": ["act:head_swivel", "act:head_tilt", "act:head_tilt_side"],
    "geom:left_eye1": [
        "act:left_eye_horizontal",
        "act:left_eye_vertical",
        "act:left_eye_torsional"
    ],
    "left_uarm1": [
        "act:left_shoulder_horizontal",
        "act:left_shoulder_abduction",
        "act:left_shoulder_internal"
    ],
    "left_larm": ["act:left_elbow"],
    "geom:left_hand1": [
        "act:left_wrist_rotation",
        "act:left_wrist_flexion",
        "act:left_wrist_ulnar",
        "act:left_fingers"
    ],
    "geom:left_upper_leg1": [
        "act:left_hip_flex",
        "act:left_hip_abduction",
        "act:left_hip_rotation"
    ],
    "geom:left_lower_leg1": ["act:left_knee"],
    "geom:left_foot2": [
        "act:left_foot_flexion",
        "act:left_foot_inversion",
        "act:left_foot_rotation",
        "act:left_toes",
    ]
}

# Expand the above mapping for the second version of MIMo.
MAPPING_MOTOR_V2 = {
    "geom:left_ffknuckle1": ["act:left_ff_side", "act:left_ff_knuckle"],
    "geom:left_ffmiddle1": ["act:left_ff_middle"],
    "geom:left_ffdistal1": ["act:left_ff_distal"],
    "geom:left_mfknuckle1": ["act:left_mf_side", "act:left_mf_knuckle"],
    "geom:left_mfmiddle1": ["act:left_mf_middle"],
    "geom:left_mfdistal1": ["act:left_mf_distal"],
    "geom:left_rfknuckle1": ["act:left_rf_side", "act:left_rf_knuckle"],
    "geom:left_rfmiddle1": ["act:left_rf_middle"],
    "geom:left_rfdistal1": ["act:left_rf_distal"],
    "geom:left_lfmetacarpal1": ["act:left_lf_meta"],
    "geom:left_lfknuckle1": ["act:left_lf_side", "act:left_lf_knuckle"],
    "geom:left_lfmiddle1": ["act:left_lf_middle"],
    "geom:left_lfdistal1": ["act:left_lf_distal"],
    "geom:left_thbase1": ["act:left_thumb_side", "act:left_thumb_add"],
    "geom:left_thhub1": ["act:left_thumb_pivot", "act:left_thumb_middle"],
    "geom:left_thdistal1": ["act:left_thumb_distal"],
    "geom:left_foot2": [
        "act:left_foot_flexion",
        "act:left_foot_inversion",
        "act:left_foot_rotation",
    ],
    "geom:left_toes1": ["act:left_toes"],
    "geom:left_big_toe1": ["act:left_big_toe"],
}


def calc_motor_gear(params: dict, defaults: dict, mimo_version: str) -> None:
    """
    Calculates the gear value for all motors based on the associated geom size
    and the ratio between gear and volume in the original model. The gear will
    be inserted into the 'params' dict.

    Arguments:
        params (dict): Growth parameters containing geom size.
        defaults (dict): Default values from the original
        mimo_version (str): Version of the MIMo model. Must be 'v1' or 'v2'.
    """

    params["motors"] = {}

    mapping = MAPPING_MOTOR.copy()
    if mimo_version == "v2":
        mapping.update(MAPPING_MOTOR_V2)

    for geom, motors in mapping.items():

        # Calculate the volume of the geom. Notice that the growth params
        # already contain values based on the age.
        type_ = defaults["geoms"][mimo_version][geom]["type"]
        size = params["geoms"][geom]["size"]
        vol = calc_volume(size, type_)

        # Get the volume of the same geom in the original model.
        base_vol = defaults["geoms"][mimo_version][geom]["vol"]

        for motor in motors:

            if motor not in defaults["motors"][mimo_version]:
                continue

            # Compute a ratio that describes the relationship between geom
            # volume and gear value in the original model. Use this ratio to
            # compute a gear value for the current geom size.
            base_gear = defaults["motors"][mimo_version][motor]["gear"]
            ratio = base_gear / base_vol
            gear = ratio * vol

            params["motors"][motor] = {"gear": gear}


def calc_geom_masses(params: dict, defaults: dict, mimo_version: str) -> None:
    """
    Calculates the mass of every geom based on the size, type, and density from
    the default model. The mass will be inserted into the 'params' dict.

    Arguments:
        params (dict): Growth parameters containing the geom size.
        defaults (dict): Default values from the original model containing
            geom type and density.
        mimo_version (str): Version of the MIMo model. Must be 'v1' or 'v2'.
    """

    default_geoms = defaults["geoms"][mimo_version]

    for geom_name, attributes in params["geoms"].items():

        # Calculate the volume.
        geom_type = default_geoms[geom_name]["type"]
        vol = calc_volume(attributes["size"], geom_type)

        # Calculate and store mass with the density.
        attributes["mass"] = vol * default_geoms[geom_name]["density"]
