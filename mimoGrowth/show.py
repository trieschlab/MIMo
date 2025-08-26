"""
Shows the growth of MIMo. This module is for visual purposes only.
"""

from mimoEnv.utils import set_joint_qpos
from mimoGrowth.growth import get_growth_params, get_version
import time
import numpy as np
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData

# Specify which ages to display on the cube in the simulation.
AGES_ON_CUBE = [0, 3, 6, 9, 12, 15, 18, 21, 24]

# Adjust how fast MIMo should grow by changing the pause time between
# increments or the age increment itself.
SLEEP_TIME = 0.02
AGE_INCR = 0.05


def update_mimo(age: float, model: MjModel, data: MjData) -> None:
    """
    Updates the MIMo model based on the given age.

    Arguments:
        age (float): The age of MIMo.
        model (MjModel): The loaded MuJoCo model.
        data (MjData): The data from the MuJoCo model.
    """

    # Get the growth parameters for the given age.
    mimo_version = get_version("mimoEnv/assets/growth.xml")
    growth_params = get_growth_params(age, mimo_version)

    # Update geoms.
    for geom_name, attributes in growth_params["geoms"].items():
        try:
            model.geom_size[model.geom(geom_name).id] = attributes["size"]
            model.geom(geom_name).pos = attributes["pos"]
        except KeyError:
            continue

    # Update bodies.
    for body_name, params in growth_params["bodies"].items():
        try:
            model.body(body_name).pos = params["pos"]
        except KeyError:
            continue

    mujoco.mj_forward(model, data)

    # Calculate correct height so MIMo stands on the ground.
    height = sum([
        -model.body("left_upper_leg").pos[2],
        -model.body("left_lower_leg").pos[2],
        -model.body("left_foot").pos[2],
        model.geom("geom:left_foot2").size[2]
    ])

    # Set the height.
    qpos = [0, 0, height, 0, 0, 0, 0]
    set_joint_qpos(model, data, "mimo_location", qpos)


def show_mimo():
    """
    Launches a MuJoCo viewer and visually shows the growth of MIMo.

    The simulation can be restarted by pressing 'strg' and the growth can be
    toggle via the 'space' key.
    """

    state = {"paused": True, "reset": False}

    def key_callback(keycode):
        if keycode == 32:  # space
            state["paused"] = not state["paused"]
        elif keycode == 341:  # strg
            state["reset"] = True

    model = mujoco.MjModel.from_xml_path("mimoEnv/assets/growth.xml")
    data = mujoco.MjData(model)

    age_months = 0
    update_mimo(age_months, model, data)

    # Store the materials needed to update the age cube.
    mat_age_cube = {}
    for age in AGES_ON_CUBE:
        mat_age_cube[age] = model.material(f"age_{age}").id

    args = {"model": model, "data": data, "key_callback": key_callback}
    with mujoco.viewer.launch_passive(**args) as viewer:
        while viewer.is_running():

            mujoco.mj_forward(model, data)
            viewer.sync()

            time.sleep(SLEEP_TIME)

            if state["reset"]:
                age_months = 0
                update_mimo(age_months, model, data)
                state["reset"], state["paused"] = False, True
                model.geom("ref_age").matid = mat_age_cube[age_months]
                continue

            if state["paused"] or age_months >= 24:
                continue

            age_months = np.round(age_months + AGE_INCR, 2)
            update_mimo(age_months, model, data)

            if age_months in mat_age_cube.keys():
                model.geom("ref_age").matid = mat_age_cube[age_months]


if __name__ == "__main__":
    show_mimo()
