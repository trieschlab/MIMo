Actuation Models
================

We currently have two models with different runtime vs accuracy trade-offs. Both approaches use the MuJoCo actuators
belonging to MIMo, but have different internal approaches to determining the output torque of these actuators. The
first, :class:`~mimoActuation.actuation.SpringDamperModel`, uses a spring-damper system to
approximate force-length and force-velocity relationships. The second, :class:`~mimoActuation.muscle.MuscleModel`,
models each actuator as two opposing, independently controllable muscles. Compared to each other, the Spring-Damper
Model is faster but less accurate, especially with regards to compliance.
In addition there is a "Positional" actuation model, which can be used to pose MIMo. The input action in this case is an
array of joint angles into which MIMo's joints are locked.


.. autosummary::
   :toctree: _autosummary
   :recursive:
   :nosignatures:
   
   mimoActuation.actuation.ActuationModel
   mimoActuation.actuation.SpringDamperModel
   mimoActuation.actuation.PositionalModel
   mimoActuation.muscle.MuscleModel
   mimoActuation.muscle_testing

mimoActuation.actuation
-----------------------

.. automodule:: mimoActuation.actuation
   :members:
   :undoc-members:
   :show-inheritance:

mimoActuation.muscle
--------------------

.. automodule:: mimoActuation.muscle
   :members:
   :undoc-members:
   :show-inheritance:

mimoActuation.muscle_testing
----------------------------

.. automodule:: mimoActuation.muscle_testing
   :members:
   :undoc-members:
   :show-inheritance:
