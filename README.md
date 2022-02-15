# MIMo

## Installation:

First install mujoco and mujoco-py following their instructions

Then clone gymTouch (https://github.com/Domattee/gymTouch), move to gymTouch/gymTouch and run `pip install -e .`

Then clone Mimo, move to the directory with setup.py and run `pip install -e .`

## Tests

There is a dummy/test environment setup in mimoEnv, as well as a very crude minimal working environment script. The script is `mimoEnv/test.py`, the environment is `MIMoTestEnv` in `mimoEnv/envs/mimo_test.py`


## Observation spaces and `done`

By default this environment follows the behaviour of the `Robot` environments in gym. This means that the `done` return value from `step` is always False, and the calling method has to figure out when to stop or reset. In addition the observation space includes two entries with the desired and the currently achieved goal (populated by `_sample_goal` and `_get_achieved_goal`).

This behaviour can be changed with two parameters during initialization of the environment. 
  1. `goals_in_observation` : If this parameter is False, the goal entries in the observation space will not be populated. Note that the space still contains these entries, but they will be size zero. By default set to True.
  2. `done_active` : If this parameter is True, `done` is True if either `_is_success` or `_is_failure` returns True. If set to False, `done` is always False. By default set to False. Note that this behaviour is defined in the `_is_done` function. If you overwrite this function you can ignore this parameter.
