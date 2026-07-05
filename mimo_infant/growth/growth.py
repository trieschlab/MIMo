"""
The entry point for adjusting the age of MIMo.

Includes:
- `log`: Helper function to log information about the growth.
- `get_version`: Helper function to return the version of MIMo.
- `get_growth_parameters`: Calculates all relevant parameters for the growth
    at the given age.
- `adjust_mimo_to_age`: Returns a scene with the updated growth parameters.

The basic workflow looks like this:
- Use the `adjust_mimo_to_age` function to create a temporary duplicate of the
provided scene where the growth parameters are updated to the given age.
- Use the returned path to load the model.
- Delete the temporary scene with the `delete_growth_scene` function. The
    function can be found within the `scene.py` script.

It is assumed that every MuJoCo scene has two <include> elements.
One that links to the meta file of MIMo and another one that links
to the actual model file. Is is important the the words *meta* and
*model* are within the file names.

It is possible to specify custom geom sizes in addition to the specified age.
Custom sizes should be provided as a dictionary with keys in the form of
`(geom_name, index)` and the values providing the geom size in meters.

Example Code:
```
from mimo_infant.growth.scene import delete_growth_scene

# Set the age of MIMo and the path to the MuJoCo scene.
scene = "path/to/the/scene.xml"
age = 2  # months

# Provide custom geom sizes.
custom = {
    # ("left_larm", 1): 0.07,  # Increase the lower arm length.
}

# Create a duplicate of your scene that includes MIMo with the specified age.
growth_scene = adjust_mimo_to_age(scene, age, custom)

# Load the MuJoCo model and data.
model = mujoco.MjModel.from_xml_path(growth_scene)
data = mujoco.MjData(model)

# Do something with the new scene.

# Delete this temporary growth scene.
delete_growth_scene(growth_scene)
```
"""

from mimo_infant.growth.utils import growth_function, mj_unit
from mimo_infant.growth.schema.schema import SCHEMA, SCHEMA_V2
from mimo_infant.growth.schema.resolve import resolve, mirror_left_elements
from mimo_infant.growth.physics import calc_geom_masses, calc_motor_gear
from mimo_infant.growth.scene import create_growth_scene
import os
import re
import copy
import json
import logging
import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

DIRNAME = os.path.dirname(__file__)

logging.basicConfig(format="%(levelname)s: %(message)s")


def log(age: float, path_scene: str) -> None:
    """
    Updates the log file with the given information.

    Arguments:
        age (float): The age of MIMo.
        path_scene (str): The path to the MuJoCo scene.
    """

    path_log = os.path.join(DIRNAME, "log.txt")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scene = os.path.basename(path_scene)
    message = f"Age of MIMo: {age:.1f} | Scene: {scene}"

    open(path_log, "a").write(f"[{timestamp}] {message}\n")

from pathlib import Path
import xml.etree.ElementTree as ET


def get_version(path: str) -> str:
    """
    Return the MIMo model version used by a scene.
    """
    root_scene = ET.parse(path).getroot()

    model_versions = []

    # First pass: MIMo model include filenames.
    for include in root_scene.findall(".//include"):
        filename = Path(include.attrib.get("file", "")).name

        if filename == "MIMo_model.xml":
            model_versions.append("v1")

        elif filename == "MIMo_modelv2.xml":
            model_versions.append("v2")

    if model_versions:
        versions = set(model_versions)

        if len(versions) > 1:
            raise ValueError(
                f"Inconsistent MIMo model includes in {path}: {sorted(versions)}"
            )

        return model_versions[0]

    # If there is no MIMo_model included infer from XML tree.
    inferred = _infer_version_from_xml_tree(root_scene)

    if inferred is not None:
        return inferred

    raise ValueError(f"Could not infer MIMo version from {path}.")


def _infer_version_from_xml_tree(root: ET.Element) -> str | None:
    """Infer MIMo version from names present in an XML tree.
    """
    xml_names = {
        elem.attrib["name"]
        for elem in root.iter()
        if "name" in elem.attrib
    }

    v1_names = (
        set(SCHEMA["geoms"])
        | set(SCHEMA["bodies"])
        | set(SCHEMA["joints"])
        | set(SCHEMA["sites"])
    )

    v2_names = (
        set(SCHEMA_V2["geoms"])
        | set(SCHEMA_V2["bodies"])
        | set(SCHEMA_V2["joints"])
        | set(SCHEMA_V2["sites"])
    )

    v2_only = v2_names - v1_names

    if xml_names & v2_only:
        return "v2"

    if xml_names & v1_names:
        return "v1"

    return None


def get_growth_params(
        age: float, mimo_version: str, custom: dict = None) -> dict:
    """
    Calculates all growth parameters for the given age and MIMo version.
    Parameters include:
    - Position, size and mass of geoms.
    - Position of bodies.
    - Gear values of motors.

    Arguments:
        age (float): The age of MIMo. Must be between 0 and 24.
        mimo_version (str): Version of MIMo. Must be 'v1' or 'v2'.
        custom (dict): Custom geom sizes for MIMo. Default is None. See the
            `adjust_mimo_to_age()` function for more details.

    Returns:
        dict: All relevant growth parameters.
    """

    # Define the path to the growth function parameters.
    path_params = os.path.join(DIRNAME, "data/params.json")

    # Load parameters for the growth functions.
    with open(path_params) as f:
        function_params = json.load(f)

    # Get the units from the measurement names.
    units = {}
    for body_part in function_params.keys():
        unit = body_part.split("_")[-1]
        name = "_".join(body_part.split("_")[:-1])
        units[name] = unit

    # Use the parameters of the approximated growth functions to
    # predict sizes for the given age.
    sizes = {}
    for body_part, params in function_params.items():
        name = "_".join(body_part.split("_")[:-1])  # Remove unit.
        sizes[name] = growth_function(age, *params)

    # Convert all sizes to the expected MuJoCo format.
    for body_part, size in sizes.items():
        measure = re.search("(circ|diam|len|breadth)", body_part).group(0)
        unit = units[body_part]
        sizes[body_part] = mj_unit(size, unit, measure)

    # Select schema based on the version of MIMo.
    schema_base = SCHEMA_V2 if mimo_version == "v2" else SCHEMA
    schema = copy.deepcopy(schema_base)

    # Update the schema based on the custom geom sizes.
    if custom:
        for (geom_name, index), custom_size in custom.items():
            if "right" in geom_name:
                logging.warning(
                    "Custom sizes should only contain left geoms. "
                    + f"Geom '{geom_name}' will be ignored."
                )
                continue
            schema["geoms"][geom_name]["size"][int(index)] = custom_size

    # Resolve the schema with the calculated sizes.
    growth_params = resolve(schema, sizes, mimo_version)

    # Load default values from original MIMo model.
    path = os.path.join(DIRNAME, "data/defaults.json")
    with open(path) as f:
        defaults = json.load(f)

    # Calculate and add mass for geoms and gear values for motors.
    calc_geom_masses(growth_params, defaults, mimo_version)
    calc_motor_gear(growth_params, defaults, mimo_version)

    # Mirror the left elements in order get the right elements.
    mirror_left_elements(growth_params)

    return growth_params


def adjust_mimo_to_age(
        age: float, path_scene: str,
        custom: dict = None, create_log: bool = False) -> str:
    """
    Creates a temporary duplicate of the provided scene where MIMo is adjusted
    to the provided age.

    Arguments:
        age (float): The age of MIMo. Possible values are between 0 and 24.
        path_scene (str): The path to the MuJoCo scene.
        custom (dict): Custom geom sizes for MIMo. Default is None.

            The dict keys need to be tuples in the form of `(geom_name, index)`
            where index refers to the position in the geom size array.
            Only left-side geoms need to be specified (e.g. "left_arm").
            The corresponding right-side geoms will be mirrored automatically.
            The dict values need to be floating numbers representing the size
            of the geom in meters. All conventions for geoms follow the
            official MuJoCo specifications.

        create_log (bool): If log files should be created. Default is False.

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

    mimo_version = get_version(path_scene)

    params = get_growth_params(age, mimo_version, custom)

    path_growth_scene = create_growth_scene(params, path_scene)

    if create_log:
        log(age, path_scene)

    return path_growth_scene
