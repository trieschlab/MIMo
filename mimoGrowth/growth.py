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

It is possible to use custom measurements in addition to the specified age.
Custom measurements should be provided in form of a dict where the keys match
the measurement names in the `mimoGrowth/data/update.py` script.

Example Code:
```
from mimoGrowth.scene import delete_growth_scene

# Set the age of MIMo and the path to the MuJoCo scene.
scene = "path/to/the/scene.xml"
age = 2  # months

# Provide custom measurements.
custom_measurements = {
    "head_circumference_cm": 35,
    "upper_arm_circumference_cm": 20,
}

# Create a duplicate of your scene that includes MIMo with the specified age.
growth_scene = adjust_mimo_to_age(scene, age, custom_measurements)

# Load the MuJoCo model and data.
model = mujoco.MjModel.from_xml_path(growth_scene)
data = mujoco.MjData(model)

# Do something with the new scene.

# Delete this temporary growth scene.
delete_growth_scene(growth_scene)
```
"""

from mimoGrowth.utils import growth_function, mj_unit
from mimoGrowth.schema.schema import SCHEMA, SCHEMA_V2
from mimoGrowth.schema.resolve import resolve, mirror_left_elements
from mimoGrowth.physics import calc_geom_masses, calc_motor_gear
from mimoGrowth.scene import create_growth_scene
import os
import re
import json
import datetime
import xml.etree.ElementTree as ET

DIRNAME = os.path.dirname(__file__)


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


def get_version(path: str) -> str:
    """
    Returns the version of the MIMo model at the given path.

    It is assumed that the MIMo model and meta files have one of the
    following names:
    - `MIMo_meta.xml` or `MIMo_metav2.xml`
    - `MIMo_model.xml` or `MIMo_modelv2.xml`

    Arguments:
        path (str): The path to the MuJoCo scene.

    Returns:
        str: The version of MIMo. Is either 'v1' or 'v2'.
    """

    # Get the path names for the model and meta file from the
    # include attributes.
    root_scene = ET.parse(path).getroot()
    includes = root_scene.findall(".//include")
    paths = [include.attrib["file"] for include in includes]

    is_v2 = ["v2" in path for path in paths]

    # Check that both paths contain the same version.
    if len(set(is_v2)) == 2:
        raise ValueError(f"Inconsistent MIMo version in {path}.")

    return "v2" if all(is_v2) else "v1"


def get_growth_params(
        age: float, mimo_version: str,
        custom_measurements: dict = None) -> dict:
    """
    Calculates all growth parameters for the given age and MIMo version.
    Parameters include:
    - Position, size and mass of geoms.
    - Position of bodies.
    - Gear values of motors.

    Arguments:
        age (float): The age of MIMo. Must be between 0 and 24.
        mimo_version (str): Version of MIMo. Must be 'v1' or 'v2'.
        custom_measurements (dict): Custom measurements for MIMo (optional).

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

    # Add the custom measurements.
    if custom_measurements is not None:
        for name, custom_size in custom_measurements.items():
            name = "_".join(name.split("_")[:-1])
            sizes[name] = custom_size

    # Convert all sizes to the expected MuJoCo format.
    for body_part, size in sizes.items():
        measure = re.search("(circ|diam|len|breadth)", body_part).group(0)
        unit = units[body_part]
        sizes[body_part] = mj_unit(size, unit, measure)

    # Load default values from original MIMo model.
    path = os.path.join(DIRNAME, "data/defaults.json")
    with open(path) as f:
        defaults = json.load(f)

    # Update the schema based on the version of MIMo.
    schema = SCHEMA_V2 if mimo_version == "v2" else SCHEMA

    # Resolve the schema with the calculated sizes.
    growth_params = resolve(schema, sizes, mimo_version)

    # Calculate and add mass.
    calc_geom_masses(growth_params, defaults)

    # Calculate and add gear values for motors.
    calc_motor_gear(growth_params, defaults, mimo_version)

    # Mirror the left elements in order get the right elements.
    mirror_left_elements(growth_params)

    return growth_params


def adjust_mimo_to_age(
        age: float, path_scene: str,
        custom_measurements: dict = None, create_log: bool = True) -> str:
    """
    Creates a temporary duplicate of the provided scene where MIMo is adjusted
    to the provided age.

    Arguments:
        age (float): The age of MIMo. Possible values are between 0 and 24.
        path_scene (str): The path to the MuJoCo scene.
        custom_measurements (dict): Custom measurements for MIMo at the given
            age. Keys need to match the measurement names in the
            `mimoGrowth/data/update.py` file. Default is none.
        create_log (bool): If log files should be created. Default is true.

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

    params = get_growth_params(age, mimo_version, custom_measurements)

    path_growth_scene = create_growth_scene(params, path_scene)

    if create_log:
        log(age, path_scene)

    return path_growth_scene
