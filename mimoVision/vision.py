import os.path

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
        self.offscreen_context = GlfwContext(offscreen=True)
        self.obs = {}

    def render_camera(self, width, height, camera_name):
        img = self.env.sim.render(width=width, height=height, camera_name=camera_name, depth=False, device_id=-1)
        return img[::-1, :, :]  # rendered image is inverted

    def get_vision_obs(self):
        # Have to manage contexts ourselves to avoid buffer reuse issues
        glfw.make_context_current(self.offscreen_context.window)
        imgs = {}
        for camera in self.camera_parameters:
            width = self.camera_parameters[camera]["width"]
            height = self.camera_parameters[camera]["height"]
            imgs[camera] = self.render_camera(width, height, camera)
        if self.env.sim._render_context_window is not None:
            glfw.make_context_current(self.env.sim._render_context_window.window)
        self.obs = imgs
        return imgs

    def save_obs_to_file(self, directory, suffix: str = ""):
        if self.obs is None or len(self.obs) == 0:
            raise RuntimeWarning("No image observations to save!")
        for camera_name in self.obs:
            file_name = camera_name + suffix + ".png"
            matplotlib.image.imsave(os.path.join(directory, file_name), self.obs[camera_name])
