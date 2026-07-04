import argparse
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

import mimo_infant.simulation

ENV_NAMES = {
    "reach": "MIMoReach-v0",
    "standup": "MIMoStandup-v0",
    "selfbody": "MIMoSelfBody-v0",
    "catch": "MIMoCatch-v0",
    "roll_over": "MIMoRollOver-v0",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENV_NAMES.keys(), default="reach")
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    env = gym.make(
        ENV_NAMES[args.env],
        render_mode=None,
    )

    raw_env = env.unwrapped
    obs, info = env.reset()

    paused = False
    reset_requested = False
    single_step_requested = False

    def key_callback(keycode):
        nonlocal paused, reset_requested, single_step_requested

        try:
            key = chr(keycode).lower()
        except ValueError:
            return

        if key == " ":
            paused = not paused

        elif key == "r":
            reset_requested = True

        elif key == "s":
            single_step_requested = True

    with mujoco.viewer.launch_passive(
        raw_env.model,
        raw_env.data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:

        viewer.sync()

        while viewer.is_running():
            loop_start = time.time()

            do_step = (not paused) or single_step_requested

            with viewer.lock():
                if reset_requested:
                    obs, info = env.reset()
                    reset_requested = False
                    single_step_requested = False

                elif do_step:
                    viewer.sync()
                    action = raw_env.data.ctrl.copy()
                    obs, reward, terminated, truncated, info = env.step(action)

                    if terminated or truncated:
                        obs, info = env.reset()

                    single_step_requested = False

            viewer.sync()

            # Keep close to real env time.
            if args.fps is not None:
                target_dt = 1.0 / args.fps
            else:
                target_dt = raw_env.dt

            sleep_time = target_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()