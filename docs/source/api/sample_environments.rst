Sample environments
===================

This section describes the code used for the experiments and demos from our paper
`MIMo: A Multi-Modal Infant Model for Studying Cognitive Development in Humans and AIs <https://ieeexplore.ieee.org/document/9962192>`_.
The learning illustration environments, :ref:`reach <sec reach>`, :ref:`standup <sec standup>`,
:ref:`self-body <sec selfbody>` and :ref:`catch <sec catch>` each involve an environment and a training script using RL
algorithms from `Stable Baselines3 <https://github.com/DLR-RM/stable-baselines3>`_. The catch environment is
based on the full hand version of MIMo, while the others use the mitten hand.
There is a simple :ref:`benchmarking <sec benchmark>` scenario in which MIMo takes random actions.
Finally there is a :ref:`demo <sec demo>` environment in a simple room with some toys, with all sensory 
modalities enabled using the default configurations.

All of the the environments register with gym under the names ``MIMoReach-v0``,
``MIMoStandup-v0``, ``MIMoSelfBody-v0``, ``MIMoCatch-v0``, ``MIMoBench-v0`` and ``MIMoShowroom-v0``.


.. contents::
   :depth: 4


.. _sec reach:

Reach Environment
-----------------

.. automodule:: mimoEnv.envs.reach
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:


.. _sec standup:

Standup Environment
-------------------

.. automodule:: mimoEnv.envs.standup
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:


.. _sec selfbody:

Self-body Environment
---------------------

.. automodule:: mimoEnv.envs.selfbody
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:


.. _sec catch:

Self-body Environment
---------------------

.. automodule:: mimoEnv.envs.catch
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:



Training script
---------------

There is also a training script for all the sample environments.

.. automodule:: mimoEnv.illustrations
   :members:
   :undoc-members:
   :show-inheritance:


.. _sec benchmark:

Benchmarking
------------

This script and the :ref:`demo <sec demo>` script use the same dummy class, but with different 
scene XMLs. For benchmarking the scene consisted of MIMo with all sensory modalities enabled
with varying configurations and a couple of objects lying on the ground. In the benchmarking 
script we take random actions after each step.

Environments
++++++++++++

.. automodule:: mimoEnv.envs.dummy
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :private-members:

Script
++++++

.. automodule:: mimoEnv.benchmark
   :members:
   :undoc-members:
   :show-inheritance:
   
   
.. _sec demo:

Demo showroom
-------------

This scenario uses the same dummy class as the benchmarking script, but replaces the basic
scene XML with a more elaborate one consisting of a square room with a number of toys.
In this scenario MIMo takes no actions at all.
   
Script
++++++

.. automodule:: mimoEnv.showroom
   :members:
   :undoc-members:
   :show-inheritance:
