Welcome to MIMo's documentation!
===================================

.. image:: imgs/showroom.png
   :width: 300
   :align: right
   :alt: MIMo with some toys

MIMo is a library for the research of cognitive development and multi-sensory learning. 
We follow the :doc:`API of the OpenAI gym <gym:content/api>` environments, using
:doc:`MuJoCo <mujoco:overview>` for the physics simulation.


.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 2
  
   guides/index.rst
   api/index.rst

MIMo is a simple, childlike robot physically simulated using MuJoCo. He has multiple sensory
modalities including touch, vision, proprioception and a vestibular system.

There are two physical models, which are identical except for the hands and feet. The base version uses
mitten-like hands with only a single finger. The feet also only have one toe. The v2 version has fully modelled
five-fingered hands based on the Adroit hand and feet with two toes. Unless stated otherwise we use the base version
for the environments.

Gym environments provide the interfaces to interact with the environments.
Sample environments and simple demo scripts can be found in the :doc:`Samples section</api/sample_environments>`.