Creating your own experiment
============================

.. contents::
   :depth: 4

In this guide we will create a new experiment using the
:ref:`standup environment <sec standup>`  as an example.
This involves creating a new scene XML and environment class.
Creating a new scene will require working with
:doc:`MuJoCo XMLs <mujoco:XMLreference>`.

This scenario will have MIMo standing up from a low crouch or sitting position.
The scene will contain MIMo and a crib like structure, with MIMos feet welded to the ground
and his hands to the crib. Episodes will have fixed length with a reward each step based on
the height of MIMos head.
We use only the proprioceptive and vestibular sensors.

The first step will be to create the XML for this scene.


The scene XML
-------------

MuJoCo allows for a modular structure by importing other XMLs. We make use of this by having
two component XMLs containing the required elements for MIMo, which are imported by the scene
XMLs. The scene XMLs are then loaded by the code.

The component XMLs are "MIMo_model.xml", which contains the kinematic tree, and
"MIMo_meta.xml", which contains the definitions of the actuators, MuJoCo sensors, textures
and so forth. These have to be split due to the XML importing process. Both are located in
``mimoEnv/assets/mimo/``.

.. highlight:: xml

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

            <texture name="crib" type="cube" builtin="flat" width="127" height="1278" rgb1="1 0.9 0.8" rgb2="1 1 1" markrgb="1 1 1"/>
            <material name="crib" texture="crib" texuniform="true"/>
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
    <include file="mimo/MIMo_meta.xml"></include>

    <worldbody>
        <geom name="floor" pos="0 0 0" size="0 0 .25" type="plane" material="matplane" condim="3"/>
        <light directional="false" diffuse=".4 .4 .4" specular="0 0 0" pos="0 0 10" dir="0 0 -1" castshadow="false"/>
        <light mode="targetbodycom" target="upper_body" directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 0 5.0" dir="0 0 -1"/>

        <!-- The location and orientation of the base model can be set using this body -->
        <body name="mimo_location" pos="0 0 .33" euler="0 0 0">
            <freejoint/>
            <include file="mimo/MIMo_model.xml"></include> <!-- Import the actual model-->
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

    <body name="crib" pos="0.078 0 0.42">
        <geom type="cylinder" material="crib" size="0.02 0.4" pos="0 0 0" euler="90 0 0"/>
        <geom type="cylinder" material="crib" size="0.01 0.2" pos="0 0 -0.2" euler="0 0 0"/>
        <geom type="cylinder" material="crib" size="0.01 0.2" pos="0 -0.2 -0.2" euler="0 0 0"/>
        <geom type="cylinder" material="crib" size="0.01 0.2" pos="0 0.2 -0.2" euler="0 0 0"/>
        <geom type="cylinder" material="crib" size="0.01 0.2" pos="0 -0.4 -0.2" euler="0 0 0"/>
        <geom type="cylinder" material="crib" size="0.01 0.2" pos="0 0.4 -0.2" euler="0 0 0"/>
        <geom type="sphere"   material="crib" size="0.022" pos="0 -0.4 0" euler="0 0 0"/>
        <geom type="sphere"   material="crib" size="0.022" pos="0 0.4 0" euler="0 0 0"/>
    </body>

There is still some trimming we can do. Since we do not use vision in this scenario we fixed
MIMos eyes and head above. However the actuators are still included in the scene and take up
resources. To disable these we replace "MIMo_meta.xml" in our scene with a copy in which we
removed those actuators, called "standup_meta.xml".

This leaves us with our finished scene XML::

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

            <texture name="crib" type="cube" builtin="flat" width="127" height="1278" rgb1="1 0.9 0.8" rgb2="1 1 1" markrgb="1 1 1"/>
            <material name="crib" texture="crib" texuniform="true"/>
        </asset>

        <!-- Import everything except the kinematic tree -->
        <include file="standup_meta.xml"></include>

        <equality>
            <weld body1="left_foot"  relpose="-0 -0.05 0 0.01 0 0 0"/>
            <weld body1="right_foot" relpose="-0 0.05 0 0.01 0 0 0"/>
            <weld body1="left_fingers"  relpose="0.1 0.1 0.45 0 -0.1 0.1 0 "/>
            <weld body1="right_fingers" relpose="0.1 -0.1 0.45 0 0.1 0.1 0 "/>
            <weld body1="head" body2="upper_body"/>
            <weld body1="left_eye" body2="head"/>
            <weld body1="right_eye" body2="head"/>
        </equality>

        <worldbody>
            <geom name="floor" pos="0 0 0" size="0 0 .25" type="plane" material="matplane" condim="3"/>
            <light directional="false" diffuse=".4 .4 .4" specular="0 0 0" pos="0 0 10" dir="0 0 -1" castshadow="false"/>
            <light mode="targetbodycom" target="upper_body" directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 0 5.0" dir="0 0 -1"/>

            <!-- The location and orientation of the base model can be set using this body -->
            <body name="mimo_location" pos="0 0 .33" euler="0 0 0">
                <freejoint/>
                <include file="mimo/MIMo_model.xml"></include> <!-- Import the actual model-->
            </body>

            <body name="crib" pos="0.078 0 0.42">
                <geom type="cylinder" material="crib" size="0.02 0.4" pos="0 0 0" euler="90 0 0"/>
                <geom type="cylinder" material="crib" size="0.01 0.2" pos="0 0 -0.2" euler="0 0 0"/>
                <geom type="cylinder" material="crib" size="0.01 0.2" pos="0 -0.2 -0.2" euler="0 0 0"/>
                <geom type="cylinder" material="crib" size="0.01 0.2" pos="0 0.2 -0.2" euler="0 0 0"/>
                <geom type="cylinder" material="crib" size="0.01 0.2" pos="0 -0.4 -0.2" euler="0 0 0"/>
                <geom type="cylinder" material="crib" size="0.01 0.2" pos="0 0.4 -0.2" euler="0 0 0"/>
                <geom type="sphere"   material="crib" size="0.022" pos="0 -0.4 0" euler="0 0 0"/>
                <geom type="sphere"   material="crib" size="0.022" pos="0 0.4 0" euler="0 0 0"/>
            </body>
        </worldbody>
    </mujoco>


The environment class
---------------------

.. highlight:: default

We start by subclassing :class:`~mimoEnv.envs.mimo_env.MIMoEnv`, adjusting the default
parameters for our experiment. The model path points to our scene XML. We don't need touch
or vision, so we disable them by passing ``None``. Proprioception and vestibular will use the
:ref:`default parameters <sec default_data>`. Since we want fixed length episodes we will
set ``done_active`` to ``False``. The parameters are simply passed through to the
parent class.

 ::

    class MIMoStandupEnv(MIMoEnv):
        def __init__(self,
                 model_path=STANDUP_XML,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 done_active=False,
                 **kwargs,
                 ):

            super().__init__(model_path=model_path,
                             proprio_params=proprio_params,
                             touch_params=touch_params,
                             vision_params=vision_params,
                             vestibular_params=vestibular_params,
                             done_active=done_active,
                             **kwargs,)

Next we need to override all the abstract functions. We will use the head height as our goal
variable::

    def get_achieved_goal(self):
        return self.data.body('head').xpos[2]

Since we want fixed length episodes and have disabled `done_active` we don't need any of the
other goal related functions and just implement them as dummy functions::

    def is_success(self, achieved_goal, desired_goal):
        return False

    def is_failure(self, achieved_goal, desired_goal):
        return False

    def is_truncated(self):
        return False

    def sample_goal(self):
        return 0.0

The only things still missing are the reward and the reset functions. The reward will consist
of a positive component based on the head height, determined in ``get_achieved_goal``, and
a penalty for large actions::

    def compute_reward(self, achieved_goal, desired_goal, info):
        quad_ctrl_cost = 0.01 * np.square(self.data.ctrl).sum()
        reward = achieved_goal - 0.2 - quad_ctrl_cost
        return reward

Finally we need to be able to reset the simulation. We reset all the positions to the state
from the XML and then slightly randomize all the joint positions, stored in the ``qpos`` array.
The first seven entries belong to the free joint between MIMo and the world, so we exclude
those from the randomization. The crib does not have joints and other joints in the
scene belong to MIMo. We then set the state with our new randomized positions and let the
simulation settle for a few timesteps::

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        qpos = self.init_crouch_position

        # set initial positions stochastically
        qpos[7:] = qpos[7:] + self.np_random.uniform(low=-0.01, high=0.01, size=len(qpos[7:]))

        # set initial velocities to zero
        qvel = np.zeros(self.data.qvel.ravel().shape)

        self.set_state(qpos, qvel)

        # perform 100 steps with no actions to stabilize initial position
        actions = np.zeros(self.action_space.shape)
        self._set_action(actions)
        mujoco.mj_step(self.model, self.data, nstep=100)

        return self._get_obs()

Finally we register our new environment with gym by adding these lines to
``mimoEnv/__init__.py``, which also lets us set our fixed episode length::

    register(id='MIMoStandup-v0',
             entry_point='mimoEnv.envs:MIMoStandupEnv',
             max_episode_steps=500,
            )

We can then create our new environment with::

    import gym
    import mimoEnv

    env = gym.make('MIMoStandup-v0')

