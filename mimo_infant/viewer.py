import argparse
import json
import os
import subprocess
import sys

import mujoco
from mujoco import viewer

from mimo_infant.growth.growth import adjust_mimo_to_age
from mimo_infant.growth.scene import delete_growth_scene

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        type=str,
        help="Path to the MuJoCo scene XML.",
    )

    parser.add_argument(
        "--age",
        type=float,
        default=None,
        help="MIMo age in months. If omitted, the original XML is used.",
    )

    args = parser.parse_args()
    xml_path = os.path.abspath(args.scene)
    generated_xml = None

    try:
        if args.age is not None:
            generated_xml = adjust_mimo_to_age(
                args.age,
                xml_path,
            )
            viewer_xml = generated_xml
        else:
            viewer_xml = xml_path

        print(f"Launching MuJoCo viewer with XML:")
        print(viewer_xml)

        model = mujoco.MjModel.from_xml_path(viewer_xml)
        data = mujoco.MjData(model)
        viewer.launch(model, data)

    finally:
        if generated_xml is not None:
            delete_growth_scene(generated_xml)


if __name__ == "__main__":
    main()