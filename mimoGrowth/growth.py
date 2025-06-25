"""
This module is the entry point for adjusting the age of MIMo.

The basic workflow looks like this:
- Use the `adjust_mimo_to_age` function to create a temporary duplicate of the
provided scene where the growth parameters are updated to the given age.
- Use the returned path to load the model.
- Delete the temporary scene with the `delete_growth_scene` function.

It is possible to use custom measurements in addition to the specified age.
Custom measurements should be provided in form of a dict where the keys match
the measurement names in the `mimoGrowth/measurements/` folder and the values
are floats in centimeters.

It is assumed that every MuJoCo scene has two <include> elements.
One that links to the meta file of MIMo and another one that links
to the actual model file. Is is important the the words *meta* and
*model* are within the file names.

The following functions should not be called directly since they will
be used by other functions:
- `calc_growth_params`
- `create_new_growth_scene`

Example Code:
```
# Set the age of MIMo and the path to the MuJoCo scene.
AGE, SCENE = 2, "path/to/the/scene.xml"

# Provide custom measurements.
custom_measurements = {
    "head_circumference": 35,
    "upper_arm_circumference": 20,
}

# Create a duplicate of your scene that
# includes MIMo with the specified age.
growth_scene = adjust_mimo_to_age(scene, age, custom_measurements)

# Do something with the new scene.
model = mujoco.MjModel.from_xml_path(growth_scene)
data = mujoco.MjData(model)

# Delete this temporary growth scene.
delete_growth_scene(growth_scene)
```
"""

from mimoGrowth.mujoco.geom_handler import calc_geom_params
from mimoGrowth.mujoco.body_handler import calc_body_params
from mimoGrowth.mujoco.motor_handler import calc_motor_params
import mimoGrowth.utils as utils
import json
import os
import datetime
import xml.etree.ElementTree as ET
import numpy as np


def adjust_mimo_to_age(
        age: float, path_scene: str, custom_measurements: dict = None,
        log: bool = True) -> str:
    """
    This function creates a temporary duplicate of the provided scene
    where the growth parameters of MIMo are adjusted to the given age.

    Arguments:
        age (float): The age of MIMo. Possible values are between 0 and 24.
        path_scene (str): The path to the MuJoCo scene.
        custom_measurements (dict): Custom measurements for MIMo. Keys need to
            match the measurement names in mimoGrowth/measurements/ and values
            are provided in centimeters.
        log (bool): If log files should be created.

    Returns:
        str: The path to the growth scene. Use this path to load the model.

    Raises:
        FileNotFoundError: If the scene path is invalid.
        ValueError: If the age is not within the valid interval.
    """

    if not os.path.exists(path_scene):
        raise FileNotFoundError(f"The path '{path_scene}' does not exist.")

    if age < 0 or age > 24:
        message = f"The Age'{age}' is invalid. Must be between 0 and 24."
        raise ValueError(message)

    params = calc_growth_params(age, path_scene, custom_measurements)

    path_growth_scene = create_growth_scene(params, path_scene)

    if log:

        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_log = os.path.join(script_dir, "log.txt")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Age of MIMo: {age:.1f} | Scene Path: {path_scene}"

        open(path_log, "a").write(f"[{timestamp}] {message}\n")

    return path_growth_scene


def calc_growth_params(
        age: float, path_scene: str, custom_measurements: dict) -> dict:
    """
    This function calculates and returns all relevant
    growth parameters. This includes:
    - Position, size and mass of geoms.
    - Position of bodies.
    - Gear values of motors.

    Arguments:
        age (float): The age of MIMo.
        path_scene (str): The path to the MuJoCo scene.
        custom_measurements (dict): Custom measurements for MIMo.

    Returns:
        dict: All relevant growth parameters.
    """

    dirname = os.path.dirname(os.path.abspath(__file__))
    path_growth_functions = os.path.join(dirname, "growth_functions.json")

    with open(path_growth_functions) as f:
        growth_functions = json.load(f)

    approx_sizes = utils.estimate_sizes(growth_functions, age)

    if custom_measurements:
        approx_sizes.update(custom_measurements)

    approx_sizes = utils.format_sizes(approx_sizes)

    base_values = utils.store_base_values(path_scene)

    params_geoms = calc_geom_params(approx_sizes, base_values)
    params_bodies = calc_body_params(params_geoms, age)
    params_motors = calc_motor_params(params_geoms, base_values)

    params = {
        "geom": params_geoms,
        "body": params_bodies,
        "motor": params_motors
    }

    return params


def create_growth_scene(growth_params: dict, path_scene: str) -> None:
    """
    This function will create duplicates of the provided scene and
    the model and meta files of MIMo. Within these duplicates, MIMo
    will have been adjusted to the specified age.

    These new files use the same name with the additional suffix '_temp' and
    will be stored in the same folders as the original files..

    Arguments:
        growth_params (dict): The growth parameters.
        path_scene (str): The path to the MuJoCo scene.
    """

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

        name = geom.attrib["name"]

        size = growth_params["geom"][name]["size"]
        geom.attrib["size"] = " ".join(np.array(size, dtype=str))

        pos = growth_params["geom"][name]["pos"]
        geom.attrib["pos"] = " ".join(np.array(pos, dtype=str))

        mass = growth_params["geom"][name]["mass"]
        geom.attrib["mass"] = str(mass)

    for body in tree_model.getroot().findall(".//body"):

        name = body.attrib["name"]

        pos = growth_params["body"][name]["pos"]
        body.attrib["pos"] = " ".join(np.array(pos, dtype=str))

    for motor in tree_meta.getroot().find("actuator").findall(".//motor"):

        name = motor.attrib["name"]

        gear = growth_params["motor"][name]["gear"]
        motor.attrib["gear"] = str(gear)

    def temp_path(path):
        return path.replace(".xml", "_temp.xml")

    tree_model.write(temp_path(path_model))
    tree_meta.write(temp_path(path_meta))

    for include in includes.values():
        include.attrib["file"] = temp_path(include.attrib["file"])

    path_growth_scene = temp_path(path_scene)
    tree_scene.write(path_growth_scene)

    return path_growth_scene


def delete_growth_scene(path_scene: str) -> None:
    """
    This function deletes the temporary growth scene and all
    associated files like the model and meta file.

    Arguments:
        path_scene (str): Path to the growth scene which will be deleted.
    """

    root_scene = ET.parse(path_scene).getroot()

    for include in root_scene.findall(".//include"):

        path_file = include.attrib["file"]
        path_file_full = os.path.join(os.path.dirname(path_scene), path_file)

        os.remove(path_file_full)

    os.remove(path_scene)
