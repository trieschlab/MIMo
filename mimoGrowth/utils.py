"""
Utility functions that are used throughout various scripts.

Includes:
- `growth_function`: Defines the function type that is used to approximate
    growth functions from the infant measurements.
- `calc_volume`: Calculates the volume of a geom based on its size and type.
- `mj_unit`: Converts one or more measurements to the expected MuJoCo format.
"""

import numpy as np


def growth_function(x: float, a: float, b: float, c: float) -> float:
    """
    Defines the function type that is used to approximate the growth functions
    from the infant measurements. Rerun the `update.py` script so that the
    changes take effect.

    By default, this is a logarithmic function. If you want to explore
    different types of approximations, simply modify the return statement
    to use other mathematical expressions. Notice that config values
    (`update.py`) may need to be adjusted.

    Example: Use `a * x ** 2 + b * x + c` if you want to try a
    quadratic function.

    Arguments:
        x (float): The input value for which the function is evaluated.
            This represents the age of MIMo and will be between 0 and 24.
        a,b,c (float): Parameters, that will modify the function.

    Returns:
        float: The result of the function evaluation at the given age.
    """

    return a * np.log(x + b) + c


def calc_volume(size: list, geom_type: str) -> float:
    """
    Calculates the volume of a geom based on its size and type.

    Arguments:
        size (list): The size of the geom.
        geom_type (str): The type of the geom. This needs to be one of the
            following: 'sphere', 'capsule', or 'box'

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


def mj_unit(nums: float | list, unit: str, measure: str) -> float | list:
    """
    Converts one ore more measurement to the format expected by MuJoCo.

    Arguments:
        nums (float | list): Numeric input value(s) to format.
        unit (str): Unit of the input value(s). Must be `cm` or `mm`.
        measure (str): Type of measurement. Must be `circ`, `len` or `diam`.

    Returns:
        (float | list): Converted value(s) in meters, scaled for MuJoCo use.
    """

    def convert(num):

        if unit == "cm":
            num /= 100
        elif unit == "mm":
            num /= 1000
        else:
            raise ValueError(f"Unknown unit: {unit}")

        if measure == "circ":
            num /= 2 * np.pi
        elif measure in ["len", "diam", "breadth"]:
            num /= 2
        else:
            raise ValueError(f"Unknown type: {measure}")

        return num

    if isinstance(nums, list):
        return [convert(num) for num in nums]

    return convert(nums)
