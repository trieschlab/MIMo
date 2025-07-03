"""..."""

from mimoGrowth.constants import MAPPING_MEASUREMENTS, URL_ANTHROKIDS, \
    MEAN_AGES_MEASUREMENTS
from mimoGrowth.utils import growth_function, calc_volume
import re
import os
import json
import requests
import argparse
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from scipy.optimize import curve_fit


DIRNAME = os.path.dirname(__file__)


def get_measurement_data(id_: float) -> pd.DataFrame:
    """..."""

    response = requests.get(URL_ANTHROKIDS.format(id=id_))

    # ...
    content = response.content.decode("utf-8").split("\r\n")[1:]
    content = [row for row in content if not re.match("^,*$", row)]
    content = [row.split(",") for row in content]

    return pd.DataFrame(content[1:], columns=content[0])


def save_as_json(data, file_name):
    """..."""

    path = os.path.join(DIRNAME, f"{file_name}.json")

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[INFO] {file_name.capitalize()} successfully updated!")


def get_geom_data():
    """..."""

    root_model = ET.parse("mimoEnv/assets/mimo/MIMo_model.xml").getroot()

    geom_data = {}
    for geom in root_model.findall(".//geom"):

        name = geom.attrib["name"]

        size = re.sub(r"\s+", " ", geom.attrib["size"]).strip()
        size = [float(num) for num in size.split(" ")]

        geom_data[name] = size

    return geom_data


def update_params():

    config = {
        "maxfev": 10000,
        "bounds": [
            (-np.inf, 0.1, -np.inf),
            (np.inf, np.inf, np.inf)
        ]
    }

    growth_functions = {}
    for body_part, ids in MAPPING_MEASUREMENTS.items():

        print(f"[INFO] Approximating function parameters for '{body_part}'")

        mean = []
        for i, id_ in enumerate(ids):

            data = get_measurement_data(id_)

            values = list(data["MEAN"]) if i == 0 else [data["MEAN"][0]]
            mean += values

        x, y = MEAN_AGES_MEASUREMENTS, mean
        params = curve_fit(growth_function, x, y, **config)[0]

        growth_functions[body_part] = list(params)

    save_as_json(growth_functions, "params")

    print("[INFO] Parameters updated successfully!")


def update_baseline():
    """..."""

    base_values = {"geom": {}, "motor": {}}

    root_model = ET.parse("mimoEnv/assets/mimo/MIMo_model.xml").getroot()
    root_meta = ET.parse("mimoEnv/assets/mimo/MIMo_meta.xml").getroot()

    for geom in root_model.findall(".//geom"):

        type_ = geom.attrib["type"]

        size = re.sub(r"\s+", " ", geom.attrib["size"]).strip()
        size = [float(num) for num in size.split(" ")]

        vol = calc_volume(size, type_)
        density = float(geom.attrib["mass"]) / vol

        base_values["geom"][geom.attrib["name"]] = {
            "type": type_,
            "size": size,
            "vol": vol,
            "density": density,
        }

    for motor in root_meta.find("actuator").findall("motor"):

        base_values["motor"][motor.attrib["name"]] = {
            "gear": float(motor.attrib["gear"])
        }

    save_as_json(base_values, "baseline")


def main():

    func_map = {
        "params": update_params,
        "baseline": update_baseline
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=func_map.keys())

    func_map[parser.parse_args().target]()


if __name__ == "__main__":
    main()
