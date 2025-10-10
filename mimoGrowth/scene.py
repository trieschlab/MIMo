"""
Functions for creating and deleting the MuJoCo XML files.

Includes:
- `create_growth_scene`: Creates a new scene, as well as model and meta files,
    where MIMo is adjusted to the specified age.
- `delete_growth_scene`: Deletes the growth scene and associated files.
"""

import os
import re
import copy
import numpy as np
import xml.etree.ElementTree as ET

def create_growth_scene(growth_params: dict, path_scene: str,
                        long_format: bool = True) -> None:
    """
    Creates a duplicate of the provided scene and the associated model and
    meta files, where MIMo is adjusted to the specified age.

    These new files use the same name with the additional suffix '_temp' and
    will be stored in the same folders as the original files.

    Arguments:
        growth_params (dict): The growth parameters.
        path_scene (str): The path to the MuJoCo scene.
    """

    tree_scene = ET.parse(path_scene)

    # Get the names of model and meta file via the include attribute.
    includes = {}
    for include in tree_scene.getroot().findall(".//include"):
        key = "model" if "model" in include.attrib["file"] else "meta"
        includes[key] = include

    # Define the model and meta file path.
    path_dir = os.path.dirname(path_scene)
    path_model = os.path.join(path_dir, includes["model"].attrib["file"])
    path_meta = os.path.join(path_dir, includes["meta"].attrib["file"])

    tree_model = ET.parse(path_model)
    tree_meta = ET.parse(path_meta)

    for geom in tree_model.getroot().findall(".//geom"):

        name = geom.attrib["name"]

        size = growth_params["geoms"][name]["size"]
        geom.attrib["size"] = " ".join(np.array(size, dtype=str))

        pos = growth_params["geoms"][name]["pos"]
        geom.attrib["pos"] = " ".join(np.array(pos, dtype=str))

        mass = growth_params["geoms"][name]["mass"]
        geom.attrib["mass"] = str(mass)

    for body in tree_model.getroot().findall(".//body"):

        name = body.attrib["name"]

        pos = growth_params["bodies"][name]["pos"]
        body.attrib["pos"] = " ".join(np.array(pos, dtype=str))

    for joint in tree_model.getroot().findall(".//joint"):

        name = joint.attrib["name"]

        if name not in growth_params["joints"]:
            continue

        pos = growth_params["joints"][name]["pos"]
        joint.attrib["pos"] = " ".join(np.array(pos, dtype=str))

    for motor in tree_meta.getroot().find("actuator").findall(".//motor"):

        name = motor.attrib["name"]

        gear = growth_params["motors"][name]["gear"]
        motor.attrib["gear"] = str(gear)

    def temp_path(path):
        return path.replace(".xml", "_temp.xml")

    # Save the new model and meta files.
    tree_model.write(temp_path(path_model))
    tree_meta.write(temp_path(path_meta))

    if long_format is False:
         
        # Update the include attributes.
        for include in includes.values():
            include.attrib["file"] = temp_path(include.attrib["file"])
        
        # Save the new scene.
        path_growth_scene = temp_path(path_scene)
        tree_scene.write(path_growth_scene)

    else:
        # Replace each <include file="..."> in the original scene text with the referenced file
            
        def _read_text(path):
            """Read file contents, strip XML decl and outer <mujoco> ... </mujoco> lines."""
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            cleaned = []
            for line in lines:
                lstrip = line.lstrip()
                if lstrip.startswith("<mujoco") or lstrip.startswith("</mujoco"):
                    continue  # skip MuJoCo root lines
                cleaned.append(line)
            txt = "".join(cleaned)

            # Remove leading XML declaration if present
            txt = re.sub(r'^\s*<\?xml[^>]*\?>\s*', '', txt, flags=re.IGNORECASE)
            return txt

        # Load the original scene *text*
        with open(path_scene, "r", encoding="utf-8") as f:
            scene_txt = f.read()

        # For each include, inline the contents of its temp file
        for inc in includes.values():
            inc_rel = inc.attrib["file"]
            inc_abs = temp_path(os.path.join(path_dir, inc_rel))

            # Regex that matches the include by its file path (self-closing or paired)
            # Keeps it simple: <include ... file="inc_rel" .../> or <include ... file="inc_rel" ...></include>
            pattern = re.compile(
                rf"""(?P<indent>[ \t]*)<include\b[^>]*\bfile=(?P<q>["']){re.escape(inc_rel)}(?P=q)[^>]*?/?>\s*(?:</include\s*>)?""",
                re.IGNORECASE
            )

            # Replacement text: raw contents of the included file (no extra indentation)
            inc_txt = _read_text(inc_abs)

            # Do the substitution everywhere this include appears
            scene_txt, n_subs = pattern.subn(lambda m: inc_txt, scene_txt)

        # Write the self-contained scene to a *_temp.xml alongside the original
        path_growth_scene = temp_path(path_scene)
        with open(path_growth_scene, "w", encoding="utf-8") as f:
            f.write(scene_txt)

        # Remove the temporary files
        os.remove(temp_path(path_model))
        os.remove(temp_path(path_meta))

    return path_growth_scene



def delete_growth_scene(growth_path_scene: str) -> None:
    """
    Deletes the temporary growth scene and all associated files like the model
    and meta file.

    Arguments:
        growth_path_scene (str): Path to the growth scene which will be deleted.
    """

    root_scene = ET.parse(growth_path_scene).getroot()

    # Remove the model and meta file.
    for include in root_scene.findall(".//include"):

        path_file = include.attrib["file"]
        path_file_full = os.path.join(os.path.dirname(growth_path_scene), path_file)

        os.remove(path_file_full)

    # Remove the scene file.
    os.remove(growth_path_scene)
