Sample environments
===================

This section describes the code used for the experiments and demos from the paper PAPERTITLE, LINK. 
The :ref:`reach <reach>`, :ref:`standup <sec standup>` and self-body experiments each involve an 
environment and a training script using the RL algorithms from StableBaselines3 LINK.
There is a simple :ref:`benchmarking <sec benchmark>` scenario in which MIMo takes random actions.
Finally there is a :ref:`demo <sec demo>` environment in a simple room with some toys, with all sensory 
modalities enabled using the default configurations. 


.. contents::
   :depth: 4

   
.. _sec reach:

Reach Experiment
----------------
   
Environment
+++++++++++

.. automodule:: mimoEnv.envs.reach
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
   
   
Training script
+++++++++++++++

.. automodule:: mimoEnv.reach
   :members:
   :undoc-members:
   :show-inheritance:


.. _sec standup:

Standup Experiment
------------------

Environment
+++++++++++

.. automodule:: mimoEnv.envs.standup
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

   
Training script
+++++++++++++++

.. automodule:: mimoEnv.standup
   :members:
   :undoc-members:
   :show-inheritance:


.. _sec benchmark:

Benchmarking
------------

This script and the :ref:`demo <sec demo>` script use the same dummy class, but with different 
scene xmls. For benchmarking the scene consisted of MIMo with all sensory modalities enabled 
with varying configurations and a couple of objects lying on the ground. In the benchmarking 
script we take random actions after each step.

Environment
+++++++++++

.. automodule:: mimoEnv.envs.dummy
   :members:
   :undoc-members:
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

This scenario uses the same dummy class as the benchmarking script, but replaces the actual 
scene XML with a more elaborate one consisting of a square room with a number of toys.
In this script MIMo takes no actions at all.
   
Script
++++++

.. automodule:: mimoEnv.showroom
   :members:
   :undoc-members:
   :show-inheritance:
