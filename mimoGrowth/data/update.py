"""
Update functions for reference data used in the growth simulation.

The data in this folder is computed once and reused across simulations.
By separating it from the main code, we keep the simulation lightweight
and efficient.

To refresh the data, run:
    `python -m data.update [params|defaults]`

Includes:
- `update_params`: Update parameters of the growth functions.
- `update_defaults`: Update the stored default values from the original model.
"""

from mimoGrowth.utils import growth_function, calc_volume
import argparse
import json
import re
import os
import requests
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from xml.etree import ElementTree as ET


DIRNAME = os.path.dirname(__file__)
DIR_MIMO = "mimoEnv/assets/mimo/"

# This URL provides the infant and children measurements for the body part
# with the given ID.
URL_ANTHROKIDS = "https://math.nist.gov/~SRessler/anthrokids/data1977/{id}.csv"

# Store the mean values for the age groups on the website. All entries except
# the last one are from the infant measurements. The last entry is the mean age
# of the first row from the children measurements. This list will be used to
# approximate the growth functions.
MEAN_AGES_MEASUREMENTS = [1, 3, 7, 10, 13.5, 17.5, 21.5, 33]

# Define which measurements we need for MIMo. The numbers indicate
# the ID's from the website for the infant and children measurements.
# Notice that some measurements have slightly different names even though
# they describe nearly the same body part.
MAPPING_MEASUREMENTS = {
    "ankle_circumference_cm": (584, 405),
    "calf_circumference_cm": (583, 397),
    "elbow_hand_length_cm": (563, 237),
    "foot_breadth_cm": (587, 421),
    "foot_length_cm": (586, 417),
    "forearm_circumference_cm": (564, 245),
    "hand_breadth_cm": (567, 265),
    "hand_length_cm": (566, 261),
    "head_circumference_cm": (557, 145),
    "hip_breadth_cm": (579, 361),  # Hip Breadth at Trochanter
    "knee_sole_length_cm": (582, 121),  # Knee Height
    "maximum_fist_breadth_cm": (569, 309),
    "mid_thigh_circumference_cm": (580, 381),  # Upper Thigh Circumference
    "middle_finger_diameter_mm": (571, 297),
    "rump_knee_length_cm": (577, 117),  # Buttock-Knee Length
    "shoulder_elbow_length_cm": (561, 221),
    "thumb_diameter_mm": (570, 281),
    "upper_arm_circumference_cm": (562, 229),
}

# Configurations for the growth function approximation. Since we use a
# logarithmic function by default, it is important to avoid the issue of log(0)
# by defining a constraint.
CONFIG = {
    "maxfev": 10000,
    "bounds": [
        (-np.inf, 0.1, -np.inf),
        (np.inf, np.inf, np.inf)
    ]
}


def store_data(data: dict, name: str) -> None:
    """
    Stores the provided dict as a JSON with the given name. The JSON will be
    stored in the same dictionary as this script.

    Arguments:
        data (dict): Data to store.
        name (str): Name of the JSON file.
    """

    path = os.path.join(DIRNAME, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def update_params() -> None:
    """
    Update the parameters for the growth functions.

    This involves:
    - Fetching the measurement data via the API.
    - Formatting the data.
    - Approximating growth functions by using the function type from the
        `utils.py` file.
    - Storing the function parameters.
    """

    params = {}
    for body_part, ids in MAPPING_MEASUREMENTS.items():

        print(f"[INFO] Approximating function parameters for '{body_part}'")

        mean = []
        for i, id_ in enumerate(ids):

            response = requests.get(URL_ANTHROKIDS.format(id=id_))

            content = response.content.decode("utf-8").split("\r\n")[1:]
            content = [row for row in content if not re.match("^,*$", row)]
            content = [row.split(",") for row in content]

            data = pd.DataFrame(content[1:], columns=content[0])

            values = list(data["MEAN"]) if i == 0 else [data["MEAN"][0]]
            mean += values

        x, y = MEAN_AGES_MEASUREMENTS, mean
        fitted_params = curve_fit(growth_function, x, y, **CONFIG)[0]

        params[body_part] = list(fitted_params)

    store_data(params, "params")

    print("[INFO] Parameters for growth functions successfully updated!")


def update_defaults() -> None:
    """
    Update the stored default values from the original MIMo models. Values from
    both MIMo versions will be stored. It is assumed that the MIMo files are
    within the `mimoEnv/assets/mimo/` directory.
    """

    defaults = {
        "geoms": {"v1": {}, "v2": {}},
        "motors": {"v1": {}, "v2": {}}
    }

    root_model = ET.parse(DIR_MIMO + "MIMo_model.xml").getroot()
    root_model_v2 = ET.parse(DIR_MIMO + "MIMo_modelv2.xml").getroot()
    root_meta = ET.parse(DIR_MIMO + "MIMo_meta.xml").getroot()
    root_meta_v2 = ET.parse(DIR_MIMO + "MIMo_metav2.xml").getroot()

    def store_geom_values(geoms, version):

        for geom in geoms:

            name = geom.attrib["name"]
            type_ = geom.attrib["type"]

            # Convert the size data type from string to an array.
            size = re.sub(r"\s+", " ", geom.attrib["size"]).strip()
            size = np.array(size.split(" "), dtype=float)

            vol = calc_volume(size, type_)
            density = float(geom.attrib["mass"]) / vol

            defaults["geoms"][version][name] = {
                "type": type_,
                "density": density,
                "vol": vol
            }

    def store_motor_values(motors, version):

        for motor in motors:

            name = motor.attrib["name"]

            defaults["motors"][version][name] = {
                "gear": float(motor.attrib["gear"])
            }

    store_geom_values(root_model.findall(".//geom"), "v1")
    store_geom_values(root_model_v2.findall(".//geom"), "v2")

    store_motor_values(root_meta.find("actuator").findall("motor"), "v1")
    store_motor_values(root_meta_v2.find("actuator").findall("motor"), "v2")

    store_data(defaults, "defaults")

    print("[INFO] Defaults from original model successfully stored!")


if __name__ == "__main__":

    func_map = {
        "params": update_params,
        "defaults": update_defaults,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("function", choices=func_map.keys())

    func_map[parser.parse_args().function]()
