Welcome to MIMo's documentation!
===================================

.. image:: imgs/showroom.png
   :width: 300
   :align: right
   :alt: MIMo with some toys

MIMo is a library for the research of cognitive development and multi-sensory learning. 
We follow the :doc:`API of Gymnasium <gym:index>` environments, using
:doc:`MuJoCo <mujoco:overview>` for the physics simulation.


.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 2
  
   guides/index.rst
   api/index.rst
   changelog.rst

MIMo is a simple, childlike robot physically simulated using MuJoCo. He has multiple sensory
modalities including touch, vision, proprioception and a vestibular system.

There are two physical models, which are identical except for the hands and feet. The base version uses
mitten-like hands with only a single finger. The feet also only have one toe. The v2 version has fully modelled
five-fingered hands based on the `Adroit hand <https://github.com/vikashplus/Adroit>`_ and feet with two toes.
There is a demo environment for the v2 version available: :class:`~mimoEnv.envs.dummy.MIMoV2DemoEnv`.

We also offer two different actuation models. See the :doc:`Actuation API</api/mimoActuation>` for more details.

The sensory modules can be configured using parameter dictionaries. Defaults exist for all of them, but note that
the parameters for the sensor modules are specific to the model used. The touch parameters for example expect that
relevant MuJoCo bodies exist in the scene and will cause an error otherwise. Since the base and the v2 version of
MIMo have different bodies they also require different touch parameters.

Gym environments provide the interfaces to interact with the environments. See the gym documentation for examples.
Sample environments and simple demo scripts can be found in the :doc:`Samples section</api/sample_environments>`.