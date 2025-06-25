from mimoGrowth.constants import MEAN_MEASUREMENT_AGES, MAPPING_MEASUREMENTS
from mimoGrowth.utils import growth_function
import re
import json
import requests
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


URL = "https://math.nist.gov/~SRessler/anthrokids/data1977/{id}.csv"


def main():

    # Use bounds for the log function to avoid the issue of log(0).
    config = {
        "maxfev": 10000,
        "bounds": [
            (-np.inf, 0.1, -np.inf),
            (np.inf, np.inf, np.inf)
        ]
    }

    growth_functions = {}

    for body_part, ids in MAPPING_MEASUREMENTS.items():

        print(f"[INFO] Approximating function for '{body_part}'")

        mean = []

        for i, id_ in enumerate(ids):

            response = requests.get(URL.format(id=id_))

            content = response.content.decode("utf-8").split("\r\n")[1:]
            content = [row for row in content if not re.match("^,*$", row)]
            content = [row.split(",") for row in content]

            df = pd.DataFrame(content[1:], columns=content[0])

            values = list(df["MEAN"]) if i == 0 else [df["MEAN"][0]]
            mean += values

        x, y = MEAN_MEASUREMENT_AGES, mean
        params = curve_fit(growth_function, x, y, **config)[0]

        growth_functions[body_part] = list(params)

    with open("mimoGrowth/growth_functions.json", "w") as f:
        json.dump(growth_functions, f, indent=4)

    print("[INFO] Growth functions updated successfully!")


if __name__ == "__main__":
    main()
