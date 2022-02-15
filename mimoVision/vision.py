import os
import sys

import mujoco_py
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
        self.viewer = None
        self._viewers = {}

        if sys.platform == "win32":
            self.offscreen_context = mujoco_py.GlfwContext(offscreen=True)
        else:
            self.offscreen_context = self._get_viewer('rgb_array').opengl_context

        self.obs = {}

    # def render_camera_legacy(self, width, height, camera_name):
    #     img = self.env.sim.render(
    #         width=width,
    #         height=height,
    #         camera_name=camera_name,
    #         depth=False,
    #         device_id=-1)
    #     return img[::-1, :, :]  # rendered image is inverted

    def render_camera(self, width, height, camera_name):
        mode = "rgb_array"
        self._get_viewer(mode).render(
            width,
            height,
            self.env.sim.model.camera_name2id(camera_name)
        )

        # window size used for old mujoco-py:
        data = self._get_viewer(mode).read_pixels(width, height, depth=False)
        # original image is upside-down, so flip it
        return data[::-1, :, :]

    def swap_context(self, window):
        glfw.make_context_current(window)

    def get_vision_obs(self):
        # Have to manage contexts ourselves to avoid buffer reuse issues
        self.swap_context(self.offscreen_context.window)

        imgs = {}
        for camera in self.camera_parameters:
            width = self.camera_parameters[camera]["width"]
            height = self.camera_parameters[camera]["height"]
            imgs[camera] = self.render_camera(width, height, camera)

        if self.env.sim._render_context_window is not None:
            self.swap_context(self.env.sim._render_context_window.window)

        self.obs = imgs
        return imgs

    def save_obs_to_file(self, directory, suffix: str = ""):
        os.makedirs(directory, exist_ok=True)
        if self.obs is None or len(self.obs) == 0:
            raise RuntimeWarning("No image observations to save!")
        for camera_name in self.obs:
            file_name = camera_name + suffix + ".png"
            matplotlib.image.imsave(os.path.join(
                directory, file_name), self.obs[camera_name])

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.env.sim)
            elif mode == "rgb_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(
                    self.env.sim, device_id=-1)
            self._viewers[mode] = self.viewer
        return self.viewer
