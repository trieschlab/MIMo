# MIMo

<img src="/docs/source/imgs/showroom.png" width="400" align="right">

MIMo is a platform for the research of the cognitive development of infants. It consists of a [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment using [MuJoCo](https://mujoco.readthedocs.io) for the physical simulation and multiple modules that can produce simulated sensory input for vision, touch, proprioception and the vestibular system.

[//]: # (See "MIMo: A Multi-Modal Infant Model for Studying Cognitive Development in Humans and AIs".)

[A full API documentation is available on ReadTheDocs.](https://mimo.readthedocs.io)

## Installation

- Clone this repository 
- `pip install -r requirements.txt` 
- `pip install -e .`

## The MIMo environment

The main class of the codebase is `MIMoEnv`. It is an openAI gym style environment, implementing all the relevant gym interfaces. It is the base class that is subclassed by all the experiment specific environments. It takes a MuJoCo XML and a series of parameter dictionaries for the sensory modalities and builds all the specific attributes, such as the observation space, from these initial inputs.

The MuJoCo XMLs defines the simulated geometries and their degrees of freedom. We have set ours up in a modular fashion to avoid duplication as much as possible. MIMos kinematic tree is defined in `mimo_model.xml` while the associated actuators and sensors are located in `mimo_meta.xml`. A scene then includes both of these files. This allows multiple scenes to share the same base model with different actuation models and ancillary objects.

The action space of the gym environment is generated automatically from the underlying MuJoCo XML. Each actuator whose name starts with 'act:' is included in the action space. Each actuator has a range from -1 to 1, with full torque in opposite directions at -1 and 1 and a linear response in between.

The observation space is a dictionary type space built automatically based on the configuration of the sensor modules. An entry 'observation' is always included and always returns relative joint positions. Enabling more sensor modules adds extra entries. For example, each camera of the vision system will store its image in a separate entry in the observation space, named after the associated camera.

### Observation spaces and `done`

By default, this environment follows the behaviour of the old `Robot` environments in gym. This means that the 'done' return value from `step` is always `False`, and the calling method has to figure out when to stop or reset. In addition, the observation space includes two entries with the desired and the currently achieved goal (populated by `sample_goal` and `get_achieved_goal`).

This behaviour can be changed with two parameters during initialization of the environment. 
  1. `goals_in_observation` : If this parameter is `False`, the goal entries will not be included in the observation space. Set to `True` by default.
  2. `done_active` : If this parameter is `True`, 'done' is `True` if either `is_success` or `is_failure` returns `True`. If set to `False`, 'done' is always `False`. By default, set to `False`. Note that this behaviour is defined in the `_is_done` function. If you overwrite this function you can ignore this parameter.

## Actuation and sensory modules

We provide two different actuation models with different trade-offs between run time and accuracy. They can be swapped out without any other adjustments to the environment.
All the sensor modules follow the same pattern. They are initialized with a MuJoCo gym environment and a dictionary of parameters and their observations can be collected by calling their `get_<modality>_obs` function. The return of this function is generally a single array containing the flattened/concatenated output. Modules can be disabled or reduced to a minimum by passing an empty dictionary. Each module also has an attribute `sensor_outputs` that stores the unflattened outputs as a dictionary. The parameter structure and workings of each module are described in more detail in the documentation.

## Sample Environments

We provide several sample environments with some simple tasks for demonstration purposes. These come with both an openAI environment in `mimoEnv/envs` as well as simple training scripts using stable-baselines3, in `mimoEnv`. These environments include:

  1. `reach` - A stripped down version where MIMo is tasked with reaching for a ball hovering in front of him. By default, only the proprioceptive sensors are used. MIMo can only move his right arm and his head is manually fixed to track the ball. The initial position of both the ball and MIMo is slightly randomized.
  2. `standup` - MIMo is tasked with standing up. At the start he is in a low crouch with his hands gripping the bars of a crib. Proprioception and the vestibular sensors are included by default.
  3. `selfbody` - MIMo is sitting on the ground and rewarded for touching a specific body part with his right arm. The target body part is randomized each episode.
  4. `catch` - A ball is dropped from a short height onto MIMo's outstretched arm. Episodes are completed successfully when MIMo holds onto the ball continuously for a full second.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
