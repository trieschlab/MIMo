Creating your own experiment
============================

.. contents::
   :depth: 4

In this guide we will create a new experiment using the
:ref:`standup environment <sec standup>`  as an example.
This involves creating a new scene XML and environment class.
Creating a new scene will require working with
:ref:`MuJoCo XMLs <mujoco:MJCF Reference>`.

This scenario will have MIMo standing up from a low crouch or sitting position.
The scene will contain MIMo and a crib like structure, with MIMos feet welded to the ground
and his hands to the crib. Episodes will have fixed length.
We use only the proprioceptive and vestibular sensors.

The first step will be to create the XML for this scene.


The scene XML
-------------

MuJoCo allows for a modular structure by importing other XMLs. We make use of this by having
two component XMLs containing the required elements for MIMo, which are imported by the scene
XMLs. The scene XMLs are loaded by the code.

The component XMLs are "MIMo_model.xml", which contains the kinematic tree, and
"MIMo_meta.xml", which contains the definitions of the actuators, MuJoCo sensors, textures
and so forth. These have to be split due to the XML importing process.

We start with a stripped down sample XML::

    <mujoco model="MIMo">

        <compiler inertiafromgeom="true" angle="degree"/>
        <option timestep="0.005" iterations="50" tolerance="1e-10" solver="Newton" jacobian="dense" cone="elliptic" impratio="1.0"/>
        <size nconmax="1000" njmax="5000" nstack="10000000" nuser_cam="3"/>

        <visual>
            <map force="0.1" zfar="30" znear="0.005"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <quality shadowsize="4096"/>
            <global offwidth="800" offheight="800"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
            <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
            <texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="1278" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>
            <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
            <material name="matgeom" texture="texgeom" texuniform="true" rgba="0.8 0.6 .4 1"/>
        </asset>

        <worldbody>
            <geom name="floor" pos="0 0 0" size="0 0 .25" type="plane" material="matplane" condim="3"/>
            <light directional="false" diffuse=".4 .4 .4" specular="0 0 0" pos="0 0 10" dir="0 0 -1" castshadow="false"/>
        </worldbody>
    </mujoco>

The ``worldbody`` element will contain the kinematic tree for the whole scene, currently just
an infinite floor plane and a light. The other elements define various parameters of the
scene, such as the maximum number of contacts in the scene and default textures. See the
MuJoCo documentation for detail.

To include MIMo we import the two component XMLs. "MIMo_model.xml" is included in the
``worldbody`` element and "MIMo_meta.xml" just above::

    <!-- Import everything except the kinematic tree -->
    <include file="MIMo_meta.xml"></include>

    <worldbody>
        <geom name="floor" pos="0 0 0" size="0 0 .25" type="plane" material="matplane" condim="3"/>
        <light directional="false" diffuse=".4 .4 .4" specular="0 0 0" pos="0 0 10" dir="0 0 -1" castshadow="false"/>
        <light mode="targetbodycom" target="upper_body" directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 0 5.0" dir="0 0 -1"/>

        <!-- The location and orientation of the base model can be set using this body -->
        <body name="mimo_location" pos="0 0 .33" euler="0 0 0">
            <freejoint/>
            <include file="MIMo_model.xml"></include> <!-- Import the actual model-->
        </body>
    </worldbody>

We also add an extra light tracking MIMos torso.

Fixing MIMos hands and feet in positions can be done by adding equality contraints::

    <equality>
        <weld body1="left_foot"  relpose="-0 -0.05 0 0.01 0 0 0"/>
        <weld body1="right_foot" relpose="-0 0.05 0 0.01 0 0 0"/>
        <weld body1="left_fingers"  relpose="0.1 0.1 0.45 0 -0.1 0.1 0 "/>
        <weld body1="right_fingers" relpose="0.1 -0.1 0.45 0 0.1 0.1 0 "/>
        <weld body1="head" body2="upper_body"/>
        <weld body1="left_eye" body2="head"/>
        <weld body1="right_eye" body2="head"/>
    </equality>

Finally we add the crib to the scene::

    Crib code?

There is still some trimming we can do. Since we do not use vision in this scenario we fixed
MIMos eyes and head above. However the actuators are still included in the scene and take up
resources. To disable these we replace "MIMo_meta.xml" in our scene with a copy in which we
removed those actuators.

The environment class
---------------------

Creating the class.

Episodes will have a fixed length, with a reward based on the height of MIMos head.
