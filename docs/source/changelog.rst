Changelog
=========

Version 1.1.0 (The Big Migration)
---------------------------------

This version moves MIMo from gym and the mujoco_py wrappers to gymnasium and
MuJoCo's own python wrappers. With this move come several breaking changes.

Swapping python wrappers means two key changes:

 -  The ``env.sim`` attribute is gone. Instead, there are now two
    attributes ``env.model`` and ``env.data`` which correspond quite closely
    to the ``sim.model`` and ``sim.data`` objects from mujoco_py
 -  Some functions are no longer available. In particular, ``model.body_name2id``
    and similar functions are not available with the MuJoCo wrappers. Instead they
    have named access, i.e. ``model.body(id).name``. See the
    :doc:`MuJoCo Documentation <mujoco:python>` for more details.

Gym has been replaced with Gymnasium throughout. The key changes due to this are:

 -  The step function now returns 5 values, ``(obs, reward, done, trunc, info)``. The
    new value 'trunc', indicates if the episode has ended for any reason other than
    reaching a terminal state, such as a time limit. As a result, any code such as
    ``if done: env.reset()`` needs to be changed to ``ìf done or trunc: env.reset()``.
 -  The reset function has also gained an extra return value. This needs to be caught
    to avoid potentially obscure unpacking errors. ``obs = env.reset()`` ->
    ``obs, _ = env.reset()`` will do.
 -  Rendering has changed significantly, depending on use case. If you wish to
    use the interactive window you can pass 'render_mode="human"' to the constructor
    and then call ``env.render()``.
    If you want to render images from multiple different cameras, or use both an
    interactive window and also render arrays (for example to save as a video), you
    should use gymnasiums MuJoCoRenderer with
    ``img = env.mujoco_renderer.render(render_mode="rgb_array", ...)``,
    similar to the old interface.

In addition to these changes there were also some adjustments to the actuation models
to allow multiple to be attached to the same environment without conflicts.

Version 1.0.0 (Initial release)
-------------------------------

First full release.

In addition to many, many small updates, this version brings two major changes that
"complete" the initial release:

 1. A new, five-fingered version of MIMo. This allows for experiments which
    require dexterous manipulation.
 2. A system to handle actuation models, with three models/implementations to start
    with. The first is the Spring-Damper model from the conference version. The
    second is a new approach in which each actuator is modeled as two opposing,
    independently controllable "muscles". Finally we have a positional model which
    allows locking MIMo into or moving him through defined poses.

The actuation systems come in a new package 'mimoActuation', which defines the
interfaces and functions that actuation models must provide, similar to the
sensory modules.
Performance was also somewhat improved.

Version 0.1.0 (Paper release)
-----------------------------

Conference paper version.