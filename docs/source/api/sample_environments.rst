Sample environments
===================

This section describes the code used for the experiments and demos from the paper PAPERTITLE, LINK. 
The :ref:`reach <reach>`, :ref:`standup <sec standup>` and self-body experiments each involve an 
environment and a training script using the RL algorithms from StableBaselines3 LINK.
A :ref:`demo <sec demo>` environment in a simple room with some toys, with all sensory modalities enabled. 
Additionarlly there is a simple :ref:`benchmarking <sec benchmark>` scenario.

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
   
   
.. _sec demo:
Demo showroom
-------------

Environment
+++++++++++

.. automodule:: mimoEnv.envs.mimo_test
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
   
Script
++++++

.. automodule:: mimoEnv.test
   :members:
   :undoc-members:
   :show-inheritance:


.. _sec benchmark:
Benchmarking
------------

Environment
+++++++++++

.. automodule:: mimoEnv.envs.mimo_test
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
   
   
