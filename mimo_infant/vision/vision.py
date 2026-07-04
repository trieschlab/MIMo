""" This module defines the vision interface and provides a simple implementation.

The interface is defined as an abstract class in :class:`~mimo_infant.vision.vision.Vision`.
A simple implementation treating each eye as a single camera is in :class:`~mimo_infant.vision.vision.SimpleVision`.

"""

import os
import matplotlib
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.mujoco import MujocoEnv
from mujoco import MjrRect
import cv2



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
            'camera_name': {'width': width, 'height': height, 'acuity': age_for_visual_acuity, 'foveation': strength_of_foveation},
            'other_camera_name': {'width': width, 'height': height, 'acuity': age_for_visual_acuity, 'foveation': strength_of_foveation},
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
        self._acuity_functions = {}
        self._foveation = {}
        for camera in camera_parameters:
            viewport = MjrRect(0, 0, camera_parameters[camera]["width"], camera_parameters[camera]["height"])
            self._viewports[camera] = viewport
            if camera_parameters[camera]["acuity"]:
                self._acuity_functions[camera] = self._get_acuity_function(
                    camera_parameters[camera]["width"], camera_parameters[camera]["height"],
                    acuity_age=camera_parameters[camera]["acuity"], fovy=camera_parameters[camera]["fovy"])
            else:
                self._acuity_functions[camera] = None
            self._foveation[camera] = camera_parameters[camera]["foveation"]

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
            img = self.env.render()
            if self._acuity_functions[camera] is not None: # Apply age-dependent visual acuity
                img = self._apply_acuity(img, camera)
            if self._foveation[camera]: # Apply foveation
                img = self._apply_foveation(img, self._foveation[camera])
            imgs[camera] = img
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

    def _apply_acuity(self, img, camera):
        """ Applies age-dependent visual acuity to the image.
        """
        new_img = np.zeros(img.shape)
        for idx in range(3):
            ff = np.fft.fft2(img[:,:,idx])
            ff_shift = np.fft.fftshift(ff)
            ff_acuity = ff_shift * self._acuity_functions[camera]
            iff = np.fft.ifftshift(ff_acuity)
            new_img[:,:,idx] = np.clip(np.abs(np.fft.ifft2(iff)), 0, 255)
        return new_img.astype(np.uint8)

    def _get_acuity_function(self, width, height, acuity_age, fovy):
        """ Computes the modulation transfer function (MTF) for visual acuity.

        We compute the MTF as a linear function between MTF=1 at 0 cycles/deg and MTF=0 at an age-dependent value interpolated
        from the values reported by Mayer et al. (1995): https://iovs.arvojournals.org/article.aspx?articleid=2179925.
        Using the size and field of view of the image, the MTF is then transformed to an array MTF(u,v) where u and v are
        the coordinates of the Fourier transform of the image. 

        Args:
            camera_parameters: A dictionary containing the configuration.

        Returns: 
            numpy.ndarray: A numpy array containing the MTF values.
        """
        ages = [1.,  1.169,  1.366,  1.596,  1.865,  2.179,  2.547,  2.976,
                3.478,  4.064,  4.75 ,  5.55 ,  6.486,  7.58 ,  8.858, 10.351,
                12.096, 14.135, 16.519, 19.304, 22.558, 26.362, 30.806, 36.]
        acuities = [0.852,  1.0005,  1.1745,  1.3795,  1.621,  1.9065,  2.2445,
                    2.645,  3.121,  3.6885,  4.367,  5.179,  6.1425,  7.1985,
                    8.218,  9.008,  9.405,  9.544,  9.6775, 10.079, 11.013,
                    12.54, 14.6785, 17.4195]
        
        if acuity_age in ages:
            acuity = acuities[ages.index(self.env.age)]
        else:
            if acuity_age < ages[0]:
                acuity = acuities[0]  # Use the first acuity value if age is smaller than all in the list
            elif acuity_age > ages[-1]:
                acuity = acuities[-1]  # Use the last acuity value if age is larger than all in the list
            else:
                for i in range(len(ages)):
                    if (ages[i] > acuity_age):
                        weight = (acuity_age - ages[i-1]) / (ages[i] - ages[i-1])
                        acuity = acuities[i-1] + (acuities[i] - acuities[i-1]) * weight
                        break

        mtf_array = np.zeros((height,width))
        fft_scale = np.max((width,height))/fovy   # Pixels to degrees
        for u in range(width):
            for v in range(height):
                mtf_array[v, u] = self._get_mtf(
                    np.sqrt(((width/2-u)/width)**2 + ((height/2-v)/height)**2) * fft_scale,
                    acuity)
        return mtf_array

    def _get_mtf(self, x, acuity):
        """ Computes the modulation transfer function (MTF) for a given spatial frequency.

        This function assumes MTF(0.1)=1, MTF(acuity) = 0.01.

        Args:
            x: Spatial frequency coverted to cycles/deg
            acuity: The frequency at which visual acuity stops.

        Returns: 
            numpy.ndarray: A numpy array containing the MTF values.
        """
        if x<0.1:
            return 1.
        b = (-2 / (1 + np.log10(acuity)))
        a = 10**b
        fx = a * x**b
        return np.clip(fx, 0.01, 0.99)
        
    def _apply_foveation(self, img, strength):
        """
        Applies foveation to the image by warping to a logarithmic mapping.

        Args:
            img: The original image to be foveated
            strength: The strength of the foveation (0: no foveation, <5: mild, >20: strong)

        Returns: 
            numpy.ndarray: A numpy array containing the foveated RGB image.
        """

        # Compute image center
        H, W = img.shape[:2]
        cx, cy = W/2.0, H/2.0

        # Compute maximum radius for mapping
        R = max(np.hypot(cx, cy), np.hypot(W-cx, cy), np.hypot(cx, H-cy), np.hypot(W-cx, H-cy))

        # Compute polar coordinates of pixels
        yy, xx = np.indices((H, W), dtype=np.float32)
        x = xx - cx
        y = yy - cy
        r_dst = np.hypot(x, y)
        theta = np.arctan2(y, x)

        # Apply logarithmic mapping to eccentricities
        if strength <= 0:
            r_src = r_dst
        else:
            logK = np.log1p(strength).astype(np.float32)
            r_src = R * (np.expm1((r_dst / R) * logK)) / strength

        # Compute new cartesian coordinates of pixels
        map_x = (cx + r_src * np.cos(theta)).astype(np.float32)
        map_y = (cy + r_src * np.sin(theta)).astype(np.float32)

        # Apply remapping
        out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        return out
            

