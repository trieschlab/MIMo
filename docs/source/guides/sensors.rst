Sensory modules
===============

All of the sensor modules follow the same pattern. They are initialized with a MuJoCo gym
environment and a dictionary of parameters and there is a single function that collects the
observations based on the scene and the initialization parameters.

This page provides a guide on working with the sensor modules, followed by a short section on
implementing new modules.

.. contents::
   :depth: 4

Setup
-----

All the sensor modules are setup in the same way. They require a Gym environment using
MuJoCo where the scene XML has already been loaded. They are initialized with two arguments:
This environment and a configuration dictionary. The form of the dictionary will be specific
to the module and may depend on the scene as well.
For example, to attach the :class:`~mimoTouch.touch.DiscreteTouch` module to some environment::

    def __init__():
        ...
        self.touch = DiscreteTouch(self, TOUCH_CONFIG)
        ...

We pass the environment instance to the touch module with a configuration dictionary.
To attach sensors to a body called "fingers", the config might look something like this::

    TOUCH_CONFIG = {
        "scales": {
            "fingers": 0.1,
        },
        "touch_function": "force_vector",
        "response_function": "spread_linear",
    }

The "scales" entry determines the distance between sensor points, while "touch_function" and
"response_function" determine the output and how it is distributed to the sensors points.
Note that the config for the touch configuration is specific to the scene, since the bodies
listed in the config have to exist in the scene.

:ref:`Default dictionaries <sec default_data>` are available for all the sensory modules, but these are designed
with MIMo in mind, and will generally not work in scenes that do not contain MIMo. They can
still be a useful guide to the structure for their module.

Collecting the sensor outputs is also very simple::

    def _get_obs():
        ...
        obs["touch"] = self.touch.get_touch_obs()
        ...

For this module specifically the output consists of a single numpy array containing the touch
sensations for all the sensor points as in the configuration dictionary.

Most of the modules come with additional functionality. For example the concrete vision
implementation has functions for saving the images to files. Check the API documentation for
the specific module.


Making your own
---------------

Any new implementations for already existing sensory modalities should meet the requirements
laid out in the associated abstract class. This means an initialization as above. Often the
abstract class already defines a stump of the initialization which provides some constraints
for the configuration. For example the :class:`~mimoProprioception.proprio.Proprioception`
base class requires that the configuration dictionary contain an entry "components" that
lists the sensory components, such as applied force, that should be included in the output.
Any implementing class should therefore use this entry and also include it as part of their
own configurations.

Additionally they must always include a function for collecting the relevant observations in
a single call with no arguments. In the modules already defined these are called
``get_<modality>_obs()``. This function should perform the whole sensory pipeline for the
implementation, calling any relevant subfunctions, etc., and finally return the finished
outputs.

All other aspects of the functionality or configuration are left to the implementing class.
For example a new proprioception class might add random noise to the outputs, with an extra
entry for the variance in the configuration and new functions for adding this noise or
processing the outputs. These will still meet the requirements as long as
``get_proprio_obs`` returns the complete and finished observations.
