"""
Functions for creating and deleting MuJoCo XML growth scenes.

Supports two scene formats:

1. Original include-based scenes
   The scene contains <include file="...model...xml"> and
   <include file="...meta...xml"> tags.

2. Expanded scenes
   The scene already contains the model, meta, actuators, sensors, etc.

In both cases, create_growth_scene(...) returns the path to a temporary grown
scene XML file, using the suffix "_temp.xml".
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import numpy as np


def create_growth_scene(
    growth_params: dict,
    path_scene: str,
    long_format: bool = True,
) -> str:
    """Create a grown duplicate of the provided scene.

    Args:
        growth_params:
            Growth parameters produced by the growth pipeline.
        path_scene:
            Path to the MuJoCo scene XML.
        long_format:
            If True, include-based scenes are converted into a self-contained
            grown scene. If False, temporary grown model/meta files are kept and
            referenced by the grown scene.

    Returns:
        Path to the temporary grown scene XML.
    """
    tree_scene = ET.parse(path_scene)
    root_scene = tree_scene.getroot()

    includes = _find_model_meta_includes(root_scene)

    if includes["model"] is None or includes["meta"] is None:
        return _create_growth_scene_from_expanded_tree(
            growth_params=growth_params,
            tree_scene=tree_scene,
            path_scene=path_scene,
        )

    return _create_growth_scene_from_includes(
        growth_params=growth_params,
        tree_scene=tree_scene,
        path_scene=path_scene,
        includes=includes,
        long_format=long_format,
    )


def delete_growth_scene(growth_path_scene: str) -> None:
    """Delete a temporary growth scene and associated temporary include files."""
    if not os.path.exists(growth_path_scene):
        return

    root_scene = ET.parse(growth_path_scene).getroot()
    scene_dir = os.path.dirname(os.path.abspath(growth_path_scene))

    for include in root_scene.findall(".//include"):
        path_file = include.attrib.get("file")
        if not path_file:
            continue

        if os.path.isabs(path_file):
            path_file_full = path_file
        else:
            path_file_full = os.path.join(scene_dir, path_file)

        # Delete generated temporary XMLs.
        if (
            os.path.exists(path_file_full)
            and os.path.basename(path_file_full).endswith("_temp.xml")
        ):
            os.remove(path_file_full)

    if (
        os.path.exists(growth_path_scene)
        and os.path.basename(growth_path_scene).endswith("_temp.xml")
    ):
        os.remove(growth_path_scene)


def _create_growth_scene_from_expanded_tree(
    *,
    growth_params: dict,
    tree_scene: ET.ElementTree,
    path_scene: str,
) -> str:
    """Apply growth directly to a self-contained expanded scene XML."""
    root_scene = tree_scene.getroot()

    _apply_growth_to_model_tree(root_scene, growth_params)
    _apply_growth_to_meta_tree(root_scene, growth_params)

    path_growth_scene = _temp_path(path_scene)
    tree_scene.write(path_growth_scene)

    return path_growth_scene


def _create_growth_scene_from_includes(
    *,
    growth_params: dict,
    tree_scene: ET.ElementTree,
    path_scene: str,
    includes: dict[str, ET.Element | None],
    long_format: bool = True,
) -> str:
    """Apply growth to classic include-based scene/model/meta files."""
    path_dir = os.path.dirname(os.path.abspath(path_scene))

    path_model = _resolve_include_path(
        includes["model"].attrib["file"],
        base_dir=path_dir,
    )
    path_meta = _resolve_include_path(
        includes["meta"].attrib["file"],
        base_dir=path_dir,
    )

    tree_model = ET.parse(path_model)
    tree_meta = ET.parse(path_meta)

    _apply_growth_to_model_tree(tree_model.getroot(), growth_params)
    _apply_growth_to_meta_tree(tree_meta.getroot(), growth_params)

    path_model_temp = _temp_path(path_model)
    path_meta_temp = _temp_path(path_meta)

    tree_model.write(path_model_temp)
    tree_meta.write(path_meta_temp)

    if not long_format:
        return _write_include_based_temp_scene(
            tree_scene=tree_scene,
            path_scene=path_scene,
            includes=includes,
            path_model_temp=path_model_temp,
            path_meta_temp=path_meta_temp,
            path_dir=path_dir,
        )

    try:
        return _write_long_format_temp_scene(
            path_scene=path_scene,
            includes=includes,
            path_model_temp=path_model_temp,
            path_meta_temp=path_meta_temp,
            path_dir=path_dir,
        )
    finally:
        if os.path.exists(path_model_temp):
            os.remove(path_model_temp)
        if os.path.exists(path_meta_temp):
            os.remove(path_meta_temp)


def _write_include_based_temp_scene(
    *,
    tree_scene: ET.ElementTree,
    path_scene: str,
    includes: dict[str, ET.Element | None],
    path_model_temp: str,
    path_meta_temp: str,
    path_dir: str,
) -> str:
    """Write a temporary scene that still references temporary model/meta XMLs."""
    model_include = includes["model"]
    meta_include = includes["meta"]

    model_include.attrib["file"] = _include_path_for_scene(
        path_model_temp,
        scene_dir=path_dir,
    )
    meta_include.attrib["file"] = _include_path_for_scene(
        path_meta_temp,
        scene_dir=path_dir,
    )

    path_growth_scene = _temp_path(path_scene)
    tree_scene.write(path_growth_scene)

    return path_growth_scene


def _write_long_format_temp_scene(
    *,
    path_scene: str,
    includes: dict[str, ET.Element | None],
    path_model_temp: str,
    path_meta_temp: str,
    path_dir: str,
) -> str:
    """Write a self-contained temporary scene by inlining grown model/meta XMLs."""
    with open(path_scene, "r", encoding="utf-8") as f:
        scene_txt = f.read()

    replacements = {
        includes["model"].attrib["file"]: path_model_temp,
        includes["meta"].attrib["file"]: path_meta_temp,
    }

    for include_file, temp_file in replacements.items():
        include_txt = _read_xml_fragment_text(temp_file)

        pattern = re.compile(
            rf"""(?P<indent>[ \t]*)<include\b[^>]*\bfile=(?P<q>["']){re.escape(include_file)}(?P=q)[^>]*?/?>\s*(?:</include\s*>)?""",
            re.IGNORECASE,
        )

        scene_txt, _ = pattern.subn(lambda _match: include_txt, scene_txt)

        # If the include path in the scene is relative but the replacement above
        # did not match due to normalization differences, also try the absolute
        # include path.
        abs_include = _resolve_include_path(include_file, base_dir=path_dir)
        if abs_include != include_file:
            pattern_abs = re.compile(
                rf"""(?P<indent>[ \t]*)<include\b[^>]*\bfile=(?P<q>["']){re.escape(abs_include)}(?P=q)[^>]*?/?>\s*(?:</include\s*>)?""",
                re.IGNORECASE,
            )
            scene_txt, _ = pattern_abs.subn(lambda _match: include_txt, scene_txt)

    path_growth_scene = _temp_path(path_scene)

    with open(path_growth_scene, "w", encoding="utf-8") as f:
        f.write(scene_txt)

    return path_growth_scene


def _apply_growth_to_model_tree(root: ET.Element, growth_params: dict) -> None:
    """Apply grown geom/body/joint/site parameters to an XML tree.
    """
    worldbody = root.find("worldbody")

    if worldbody is None:
        # This can happen when the function is called on a standalone model
        # fragment whose root is already effectively the model body subtree.
        search_root = root
    else:
        search_root = worldbody

    for geom in search_root.findall(".//geom"):
        name = geom.attrib.get("name")
        if name not in growth_params.get("geoms", {}):
            continue

        geom_params = growth_params["geoms"][name]

        if "size" in geom_params:
            geom.attrib["size"] = _array_attr(geom_params["size"])

        if "pos" in geom_params:
            geom.attrib["pos"] = _array_attr(geom_params["pos"])

        if "mass" in geom_params:
            geom.attrib["mass"] = str(geom_params["mass"])

    for body in search_root.findall(".//body"):
        name = body.attrib.get("name")
        if name not in growth_params.get("bodies", {}):
            continue

        body_params = growth_params["bodies"][name]

        if "pos" in body_params:
            body.attrib["pos"] = _array_attr(body_params["pos"])

    # Important: only physical joints under <worldbody>, not equalities or other.
    for joint in search_root.findall(".//joint"):
        name = joint.attrib.get("name")
        if name not in growth_params.get("joints", {}):
            continue

        joint_params = growth_params["joints"][name]

        if "pos" in joint_params:
            joint.attrib["pos"] = _array_attr(joint_params["pos"])

    for site in search_root.findall(".//site"):
        name = site.attrib.get("name")
        if name not in growth_params.get("sites", {}):
            continue

        site_params = growth_params["sites"][name]

        if "pos" in site_params:
            site.attrib["pos"] = _array_attr(site_params["pos"])
            

def _apply_growth_to_meta_tree(root: ET.Element, growth_params: dict) -> None:
    """Apply grown motor gear values to an XML tree.
    """
    motor_params = growth_params.get("motors", {})

    for motor in root.findall(".//motor"):
        name = motor.attrib.get("name")
        if name not in motor_params:
            continue

        gear = motor_params[name].get("gear")
        if gear is None:
            continue

        motor.attrib["gear"] = str(gear)


def _find_model_meta_includes(root_scene: ET.Element) -> dict[str, ET.Element | None]:
    """Find model/meta include tags if present.
    """
    includes: dict[str, ET.Element | None] = {
        "model": None,
        "meta": None,
    }

    for include in root_scene.findall(".//include"):
        file_attr = include.attrib.get("file", "")
        filename = os.path.basename(file_attr).lower()

        if not filename.endswith(".xml"):
            continue

        if "meta" in filename:
            includes["meta"] = include
            continue

        if (
            "model" in filename
            or filename in {"mimo.xml", "mimo_model.xml", "mimo_modelv2.xml"}
        ):
            includes["model"] = include
            continue

    return includes


def _resolve_include_path(path_file: str, *, base_dir: str) -> str:
    """Resolve an include path relative to a scene directory."""
    if os.path.isabs(path_file):
        return path_file

    return os.path.abspath(os.path.join(base_dir, path_file))


def _include_path_for_scene(path_file: str, *, scene_dir: str) -> str:
    """Return a path suitable for writing into an include tag.
    """
    try:
        return os.path.relpath(path_file, scene_dir)
    except ValueError:
        return path_file


def _read_xml_fragment_text(path: str) -> str:
    """Read XML contents without XML declaration or outer <mujoco> tags."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned = []

    for line in lines:
        stripped = line.lstrip()

        if stripped.startswith("<?xml"):
            continue

        if stripped.startswith("<mujoco") or stripped.startswith("</mujoco"):
            continue

        cleaned.append(line)

    txt = "".join(cleaned)
    txt = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", txt, flags=re.IGNORECASE)

    return txt


def _array_attr(values) -> str:
    """Convert a vector-like value to a MuJoCo XML attribute string."""
    return " ".join(np.array(values, dtype=str))


def _temp_path(path: str) -> str:
    """Return the temporary growth path for an XML file."""
    if path.endswith(".xml"):
        return path[:-4] + "_temp.xml"

    return path + "_temp.xml"