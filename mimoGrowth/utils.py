""" This module store utility and helper functions. """

from mimoGrowth.constants import RATIOS_MIMO_GEOMS
import re
import os
import numpy as np
import xml.etree.ElementTree as ET


def growth_function(x, a, b, c) -> float:
    """
    This function represents the standard form of the growth functions.

    By default, this is a logarithmic function. If you want to explore
    different types of approximations, simply modify the return statement
    to use other mathematical expressions (e.g., a quadratic function).

    Example: Use `a * x ** 2 + b * x + c` if you want to try a
    quadratic function.

    Notice that the bounds in `approximate_growth_functions` should be
    changed accordingly.

    Arguments:
        x (float): The input value for which the function is evaluated.
            This represents the age of MIMo and will be between 0 and 24.
        a, b, c (float): Parameters, that will modify the function.

    Returns:
        float: The result of the function evaluation at the given `x`.
    """

    return a * np.log(x + b) + c


def estimate_sizes(functions: dict, age: float) -> dict:
    """
    This function uses the approximated functions and the given age
    to estimate sizes for each body part.

    Arguments:
        functions (dict): The growth functions for all body parts.
        age (float): The age of MIMo.

    Returns:
        dict: The predicted size for every body part at the given age.
    """

    sizes = {}
    for body_part, params in functions.items():
        sizes[body_part] = growth_function(age, *params)

    return sizes


def format_sizes(sizes: dict) -> dict:
    """
    This function will format the estimated sizes.
    Specifically, this means:
    - Converting units to MuJoCo standards
    - Group measurements so they can be associated with a geom
    - Applying ratios

    This list describes the high-level body parts and the
    corresponding measurements:
    - head      : Head Circumference
    - upper_arm : [Upper Arm Circumference, Shoulder Elbow Length]
    - lower_arm : [Forearm Circumference, Elbow Hand Length - Hand Length]
    - hand      : [Hand Length, Hand Breadth, Maximum Fist Breadth]
    - torso     : Hip Breadth
    - upper_leg : [Mid Thigh Circumference, Rump Knee Length]
    - lower_leg : [Calf Circumference, Ankle Circumference, Knee Sole Length]
    - foot      : [Foot Length, Foot Breadth]

    Arguments:
        sizes (dict): The estimated sizes for all body parts.

    Returns:
        dict: The formatted sizes for all body parts.
    """

    # Use meter as unit and convert circumference to radius or
    # split lengths in half. MuJoCo expects these units.
    for body_part, meas in sizes.items():
        sizes[body_part] = np.array(meas) / 100
        sizes[body_part] /= 2 * np.pi if "circum" in body_part else 2

    # Group the measurements. This will make later calculations easier.
    # Notice that for some body parts we need to subtract the radius from the
    # length since MuJoCo expects the half-length only of the cylinder part.
    sizes = {
        "head": [sizes["head_circumference"]],
        "upper_arm": [
            sizes["upper_arm_circumference"],
            sizes["shoulder_elbow_length"] - sizes["upper_arm_circumference"]
        ],
        "lower_arm": [
            sizes["forearm_circumference"],
            (
                sizes["elbow_hand_length"] -
                sizes["hand_length"] -
                sizes["forearm_circumference"]
            )
        ],
        "hand": [
            sizes["hand_length"],
            sizes["hand_breadth"],
            sizes["maximum_fist_breadth"]
        ],
        # For the torso we need to duplicate the size by five
        # since the whole torso is made up of five capsules.
        # Each capsule will be tweaked a little by the ratio later.
        "torso": np.repeat(sizes["hip_breadth"], 5),
        "upper_leg": [
            sizes["mid_thigh_circumference"],
            sizes["rump_knee_length"] - sizes["mid_thigh_circumference"]
        ],
        "lower_leg": [
            sizes["calf_circumference"],
            sizes["ankle_circumference"],
            (
                sizes["knee_sole_length"] -
                sizes["calf_circumference"] / 2 -
                sizes["ankle_circumference"] / 2
            )
        ],
        "foot": [sizes["foot_length"], sizes["foot_breadth"]]
    }

    for body_part in sizes.keys():
        sizes[body_part] *= np.array(RATIOS_MIMO_GEOMS[body_part])

    return sizes


def calc_volume(size: list, geom_type: str) -> float:
    """
    This function returns the volume based on the size and type of a geom.

    Arguments:
        size (list): The size of the geom.
        geom_type (str): The type of the geom. This needs to be one of the
        following: 'sphere', 'capsule' or 'box'

    Returns:
        float: The volume of the geom.

    Raises:
        ValueError: If the geom type is invalid.
    """

    if geom_type == "sphere":
        vol = (4 / 3) * np.pi * size[0] ** 3

    elif geom_type == "capsule":
        vol = (4 / 3) * np.pi * size[0] ** 3
        vol += np.pi * size[0] ** 2 * size[1] * 2

    elif geom_type == "box":
        vol = np.prod(size) * 8

    elif geom_type == "cylinder":
        vol = np.pi * size[0] ** 2 * size[1] * 2

    else:
        raise ValueError(f"Unknown geom type '{geom_type}'.")

    return vol


def store_base_values(path_scene: str) -> None:
    """
    This function stores relevant values of the original MIMo model before
    the age is changed.

    Arguments:
        path_scene (str): The path to the MuJoCo scene.

    Returns:
        dict: All relevant values of MIMo.
    """

    base_values = {"geom": {}, "motor": {}}

    tree_scene = ET.parse(path_scene)

    includes = {}
    for include in tree_scene.getroot().findall(".//include"):
        key = "model" if "model" in include.attrib["file"] else "meta"
        includes[key] = include

    path_dir = os.path.dirname(path_scene)
    path_model = os.path.join(path_dir, includes["model"].attrib["file"])
    path_meta = os.path.join(path_dir, includes["meta"].attrib["file"])

    tree_model = ET.parse(path_model)
    tree_meta = ET.parse(path_meta)

    for geom in tree_model.getroot().findall(".//geom"):

        type_ = geom.attrib["type"]

        size = re.sub(r"\s+", " ", geom.attrib["size"]).strip()
        size = np.array(size.split(" "), dtype=float)

        vol = calc_volume(size, type_)
        density = float(geom.attrib["mass"]) / vol

        base_values["geom"][geom.attrib["name"]] = {
            "type": type_,
            "size": size,
            "vol": vol,
            "density": density,
        }

    for motor in tree_meta.getroot().find("actuator").findall("motor"):

        base_values["motor"][motor.attrib["name"]] = {
            "gear": float(motor.attrib["gear"])
        }

    return base_values
