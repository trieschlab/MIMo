Sample environments
===================

This section describes the code used for the experiments and demos from the paper PAPERTITLE, LINK. 
The :ref:`reach <sec reach>`, :ref:`standup <sec standup>` and :ref:`self-body <sec selfbody>`
experiments each involve an environment and a training script using the RL algorithms from
`Stable Baselines3 <https://github.com/DLR-RM/stable-baselines3>`. The scripts are practically
identical for all experiments.
There is a simple :ref:`benchmarking <sec benchmark>` scenario in which MIMo takes random actions.
Finally there is a :ref:`demo <sec demo>` environment in a simple room with some toys, with all sensory 
modalities enabled using the default configurations. 


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


Training scripts
----------------

The training scripts for all the sample environments are functionally identical, so we
document only one of them in detail.

Summary
+++++++

.. autosummary::
   :toctree: _autosummary

   mimoEnv.reach
   mimoEnv.standup
   mimoEnv.selfbody

Documentation
+++++++++++++

.. automodule:: mimoEnv.reach
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

This scenario uses the same dummy class as the benchmarking script, but replaces the basic
scene XML with a more elaborate one consisting of a square room with a number of toys.
In this scenario MIMo takes no actions at all.
   
Script
++++++

.. automodule:: mimoEnv.showroom
   :members:
   :undoc-members:
   :show-inheritance:
