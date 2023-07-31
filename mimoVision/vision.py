""" This module defines the vision interface and provides a simple implementation.

The interface is defined as an abstract class in :class:`~mimoVision.vision.Vision`.
A simple implementation treating each eye as a single camera is in :class:`~mimoVision.vision.SimpleVision`.

"""

import os
import matplotlib
from typing import Dict
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from mujoco import MjrRect



class Vision:
    """ Abstract base class for vision.

    This class defines the functions that all implementing classes must provide.
    The constructor takes two arguments: `env`, which is the environment we are working with, and `camera_parameters`,
    which can be used to supply implementation specific parameters.

    There is only one function that implementations must provide:
    :meth:`.get_vision_obs` should produce the vision outputs that will be returned to the environment. These outputs
    should also be stored in :attr:`.sensor_outputs`.

    Attributes:
        env (MujocoEnv): The environment to which this module will be attached
        camera_parameters: A dictionary containing the configuration. The exact from will depend on the specific
            implementation.
        sensor_outputs: A dictionary containing the outputs produced by the sensors. Shape will depend on the specific
            implementation. This should be populated by :meth:`.get_vision_obs`

    """
    def __init__(self, env, camera_parameters):
        self.env = env
        self.camera_parameters = camera_parameters
        self.sensor_outputs = {}

    def get_vision_obs(self):
        """ Produces the current vision output.

        This function should perform the whole sensory pipeline and return the vision output as defined in
        :attr:`.camera_parameters`. Exact return value and functionality will depend on the implementation, but should
        always be a dictionary containing images as values.

        Returns:
            Dict[str, np.ndarray]: A dictionary of numpy arrays with the output images.

        """
        raise NotImplementedError


class SimpleVision(Vision):
    """ A simple vision system with one camera for each output.

    The output is simply one RGB image for each camera in the configuration. The constructor takes two arguments: `env`,
    which is the environment we are working with, and `camera_parameters`, which provides the configuration for the
    vision system.
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

    """
    def __init__(self, env, camera_parameters):
        """ Constructor.

        Args:
            env: The environment to which this module should be attached
            camera_parameters: A dictionary containing the configuration.

        """
        super().__init__(env, camera_parameters)
        self._viewports = {}
        for camera in camera_parameters:
            viewport = MjrRect(0, 0, camera_parameters[camera]["width"], camera_parameters[camera]["height"])
            self._viewports[camera] = viewport

    def get_vision_obs(self):
        """ Produces the current vision output.

        This function renders each camera with the resolution as defined in :attr:`.camera_parameters` using an
        off-screen render context. The images are also stored in :attr:`.sensor_outputs` under the name of the
        associated camera.

        Returns:
            Dict[str, np.ndarray]: A dictionary with camera names as keys and the corresponding rendered images as
            values.
        """
        # We have to cycle render modes, camera names, camera ids and viewport sizes
        old_mode = self.env.render_mode
        old_cam_name = self.env.camera_name
        old_cam_id = self.env.camera_id

        # Ensure that viewer is initialized
        if not self.env.mujoco_renderer._viewers.get("rgb_array"):
            self.env.mujoco_renderer.render(render_mode="rgb_array")

        rgb_viewer = self.env.mujoco_renderer._viewers["rgb_array"]
        old_viewport = rgb_viewer.viewport

        self.env.render_mode = "rgb_array"
        self.env.camera_id = None

        imgs = {}
        for camera in self.camera_parameters:
            self.env.camera_name = camera
            rgb_viewer.viewport = self._viewports[camera]
            imgs[camera] = self.env.render()
        self.sensor_outputs = imgs

        self.env.render_mode = old_mode
        self.env.camera_name = old_cam_name
        self.env.camera_id = old_cam_id
        rgb_viewer.viewport = old_viewport

        return imgs

    def save_obs_to_file(self, directory, suffix=""):
        """ Saves the output images to file.

        Everytime this function is called all images in :attr:`.sensor_outputs` are saved to separate files in
        `directory`. The filename is determined by the camera name and `suffix`. Saving large images takes a long time!

        Args:
            directory (str): The output directory. It will be created if it does not already exist.
            suffix (str): Optional file suffix. Useful for a step counter. Empty by default.
        """
        os.makedirs(directory, exist_ok=True)
        if self.sensor_outputs is None or len(self.sensor_outputs) == 0:
            raise RuntimeWarning("No image observations to save!")
        for camera_name in self.sensor_outputs:
            file_name = camera_name + suffix + ".png"
            matplotlib.image.imsave(os.path.join(
                directory, file_name), self.sensor_outputs[camera_name], vmin=0.0, vmax=1.0)
