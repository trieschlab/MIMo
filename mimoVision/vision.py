import os
import sys

import mujoco_py
from mujoco_py import GlfwContext
import matplotlib
import glfw


class Vision:
    def __init__(self, env, camera_parameters):
        self.env = env
        self.camera_parameters = camera_parameters

    def render_camera(self, width, height, camera_name):
        raise NotImplementedError

    def get_vision_obs(self):
        raise NotImplementedError


class SimpleVision(Vision):
    def __init__(self, env, camera_parameters):
        super().__init__(env, camera_parameters)
        self.on_windows = sys.platform == 'win32'  # Hacky check to try and intercept GLEW errors
        if self.on_windows:
            self.offscreen_context = GlfwContext(offscreen=True)
        else:
            self.offscreen_context = mujoco_py.MjRenderContextOffscreen(self.env.sim, -1)
        self.obs = {}

    def render_camera(self, width, height, camera_name):
        # Have to handle the contexts yourself if you are using this function on windows
        img = self.env.sim.render(width=width, height=height, camera_name=camera_name, depth=False, device_id=-1)
        return img[::-1, :, :]  # rendered image is inverted

    def swap_context(self, window):
        glfw.make_context_current(window)

    def get_vision_obs(self):
        # Have to manage contexts ourselves to avoid buffer reuse issues
        if self.on_windows:
            self.swap_context(self.offscreen_context.window)
        imgs = {}
        for camera in self.camera_parameters:
            width = self.camera_parameters[camera]["width"]
            height = self.camera_parameters[camera]["height"]
            imgs[camera] = self.render_camera(width, height, camera)
        if self.env.sim._render_context_window is not None and self.on_windows:
            self.swap_context(self.env.sim._render_context_window.window)
        self.obs = imgs
        return imgs

    def save_obs_to_file(self, directory, suffix: str = ""):
        if self.obs is None or len(self.obs) == 0:
            raise RuntimeWarning("No image observations to save!")
        for camera_name in self.obs:
            file_name = camera_name + suffix + ".png"
            matplotlib.image.imsave(os.path.join(directory, file_name), self.obs[camera_name])
