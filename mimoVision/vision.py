""" This module defines the vision interface and provides a simple implementation.

The interface is defined as an abstract class in :class:`~mimoVision.vision.Vision`.
A simple implementation treating each eye as a single camera is in :class:`~mimoVision.vision.SimpleVision`.

"""

import os
import sys

import mujoco_py
import matplotlib
import glfw


class Vision:
    """ Abstract base class for vision.

    This class defines the functions that all implementing classes must provide.
    :meth:`.get_vision_obs` should produce the vision outputs that will be returned to the environment. These outputs
    should also be stored in :attr:`.sensor_outputs`. :meth:`.render_camera` can be used to render any camera in the
    scene.

    Attributes:
        env: The environment to which this module will be attached
        camera_parameters: A dictionary containing the configuration. The exact from will depend on the specific
            implementation.
        sensor_outputs: A dictionary containing the outputs produced by the sensors. Shape will depend on the specific
            implementation. This should be populated by :meth:`.get_vision_obs`

    """

    def __init__(self, env, camera_parameters):
        self.env = env
        self.camera_parameters = camera_parameters
        self.sensor_outputs = {}

    def render_camera(self, width: int, height: int, camera_name: str):
        """ Renders images of a given camera.

        Given the name of a camera in the scene, renders an image with it using the resolution provided by `width` and
        `height`. The vertical field of view is defined in the scene xml, with the horizontal field of view determined
        by the rendering resolution.

        Args:
            width: The width of the output image
            height: The height of the output image
            camera_name: The name of the camera that will be used for rendering.

        Returns:
            A numpy array with the containing the output image.

        """
        raise NotImplementedError

    def get_vision_obs(self):
        """ Produces the current vision output.

        This function should perform the whole sensory pipeline and return the vision output as defined in
        :attr:`.camera_parameters`. Exact return value and functionality will depend on the implementation, but should
        always be a dictionary containing images as values.

        Returns:
            A dictionary of numpy arrays with the output images.

        """
        raise NotImplementedError


class SimpleVision(Vision):
    """ A simple vision system with one camera for each output.

    The output is simply one RGB image for each camera in the configuration. The constructor takes two arguments: `env`,
    which is the environment we are working with, and `camera_parameters`.
    The parameter `camera_parameters` should be a dictionary with the following structure::

        {
            'camera_name': {'width': width, 'height': height},
            'other_camera_name': {'width': width, 'height': height},
        }

    The default MIMo model has two cameras, one in each eye, named `eye_left` and `eye_right`. Note that the cameras in
    the dictionary must exist in the scene xml or errors will occur!

    Attributes:
        env: The environment to which this module should be attached
        camera_parameters: A dictionary containing the configuration.
        sensor_outputs: A dictionary containing the outputs produced by the sensors. This is populated by
            :meth:`.get_vision_obs`
        viewer: The currently active render context.

    """
    def __init__(self, env, camera_parameters):
        """ Constructor.

        Args:
            env: The environment to which this module should be attached
            camera_parameters: A dictionary containing the configuration.

        """
        super().__init__(env, camera_parameters)
        self.viewer = None
        self._viewers = {}

        if sys.platform != "darwin":
            self.offscreen_context = mujoco_py.GlfwContext(offscreen=True)
        else:
            self.offscreen_context = self._get_viewer('rgb_array').opengl_context

    def render_camera(self, width: int, height: int, camera_name: str):
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

    def _swap_context(self, window):
        """ Swaps the current render context to 'window'

        Args:
            window: The new render context.

        """
        glfw.make_context_current(window)

    def get_vision_obs(self):
        """ Produces the current vision output.

        This function renders each camera with the resolution as defined in :attr:`.camera_parameters` using an
        offscreen render context. The images are stored in :attr:`.sensor_outputs` under the name of the associated
        camera. Uses :meth:`.render_camera`.

        Returns:
            A dictionary of numpy arrays. Keys are camera names and the values are the corresponding images.

        """
        # Have to manage contexts ourselves to avoid buffer reuse issues
        if self.env.sim._render_context_window is not None:
            self._swap_context(self.offscreen_context.window)

        imgs = {}
        for camera in self.camera_parameters:
            width = self.camera_parameters[camera]["width"]
            height = self.camera_parameters[camera]["height"]
            imgs[camera] = self.render_camera(width, height, camera)

        if self.env.sim._render_context_window is not None:
            self._swap_context(self.env.sim._render_context_window.window)

        self.sensor_outputs = imgs
        return imgs

    def save_obs_to_file(self, directory: str, suffix: str = ""):
        """ Saves the output images to file.

        Everytime this function is called all images in :attr:`.sensor_outputs` are saved to separate files in
        `directory`. The filename is determined by the camera name and `suffix`. Saving large images takes a long time!

        Args:
            directory: The output directory. It will be created if it does not already exist.
            suffix: Optional file suffix. Useful for a step counter.

        """
        os.makedirs(directory, exist_ok=True)
        if self.sensor_outputs is None or len(self.sensor_outputs) == 0:
            raise RuntimeWarning("No image observations to save!")
        for camera_name in self.sensor_outputs:
            file_name = camera_name + suffix + ".png"
            matplotlib.image.imsave(os.path.join(
                directory, file_name), self.sensor_outputs[camera_name])

    def _get_viewer(self, mode: str):
        """ Handles render contexts.

        Args:
            mode: One of 'human' or 'rgb_array'. If 'rgb_array' an offscreen render context is used, otherwise we render
            to an interactive viewer window.

        """
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.env.sim)
            elif mode == "rgb_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(
                    self.env.sim, device_id=-1)
            self._viewers[mode] = self.viewer
        return self.viewer
