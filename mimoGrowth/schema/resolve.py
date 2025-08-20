"""
Functions to resolve the schema.

Includes:
- `apply_op`: Helper function to resolve operators.
- `mirror_left`: Helper function to create the right elements.
- `resolve`: Main function to resolve the schema.
"""

import copy
import numpy as np
from mimoGrowth.schema.schema import SCHEMA, SCHEMA_V2


def apply_op(op: str, args: list) -> float:
    """
    Applies the given operator to the list of arguments.

    Arguments:
        op (str): The operator. Must be one of: 'neg', 'add', 'sub', 'mul',
            'div', or 'mean'
        args (list): The list of arguments.

    Returns:
        float: The result of applying the operator to the arguments.

    Raises:
        ValueError: If the operator is unknown.
    """

    if op == "neg":
        return -args[0]
    if op == "add":
        return np.sum(args)
    if op == "sub":
        return np.subtract.reduce(args)
    if op == "mul":
        return np.prod(args)
    if op == "div":
        return np.divide.reduce(args)
    if op == "mean":
        return np.mean(args)

    raise ValueError(f"Unknown op: {op}")


def mirror_left_elements(params: dict) -> None:
    """
    Mirrors all left elements from the growth parameters to get their right
    counterparts.

    Arguments:
        params (dict): The growth parameters (resolved schema).
    """

    # Iterate over all geoms, bodies, and motors.
    for element_type, elements in params.items():

        mirrored_elements = {}
        for name, attributes in elements.items():

            if "left" in name:

                mirrored_name = name.replace("left", "right")
                mirrored_attrs = copy.deepcopy(attributes)

                # Negate the y-pos for bodies (left/right movement) so that
                # the body is correctly placed.
                if element_type == "bodies" and "pos" in mirrored_attrs:
                    mirrored_attrs["pos"][1] *= -1

                mirrored_elements[mirrored_name] = mirrored_attrs

        elements.update(mirrored_elements)


def resolve(obj: dict, sizes: dict, mimo_version: str) -> None:
    """
    Resolve the schema in-place by recursively iterating through the schema.

    Arguments:
        obj (dict): Current object to resolve.
        sizes (dict): Approximated measurement sizes for all body parts.
        mimo_version (str): Version of the MIMo model. Must be 'v1' or 'v2'.
    """

    # Return the predicted measurement size.
    if isinstance(obj, str):
        return sizes[obj]

    # Apply operators.
    if isinstance(obj, dict) and "$op" in obj:
        resolved_args = [
            resolve(arg, sizes, mimo_version) for arg in obj["args"]
        ]
        return apply_op(obj["$op"], resolved_args)

    # Get the referenced object.
    if isinstance(obj, dict) and "$ref" in obj:
        node = SCHEMA_V2 if mimo_version == "v2" else SCHEMA
        for key in obj["$ref"]:
            node = node[key]
        return resolve(node, sizes, mimo_version)

    # Call this function for every item in a list and return a padded array
    # since MuJoCo needs to have all arrays with length three.
    if isinstance(obj, list):
        resolved_list = [resolve(item, sizes, mimo_version) for item in obj]
        return np.pad(resolved_list, (0, 3 - len(resolved_list)))

    # Call this function for every key-value pair in the dictionary.
    if isinstance(obj, dict):
        return {
            key: resolve(val, sizes, mimo_version) for key, val in obj.items()
        }

    return obj
