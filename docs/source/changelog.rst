Changelog
=========

Version MuJoCo (Proper Versions soon)
-------------------------------------

This version moves MIMo from gym and the mujoco_py wrappers to gymnasium and
MuJoCo's own python wrappers. With this change come several breaking changes:

 - The step function now returns 5 values, ``(obs, reward, done, trunc, info)``. The
   new value 'trunc', indicates if the episode has ended for any reason other than
   reaching a terminal state, such as a time limit. As a result, any code such as
   ``if done: env.reset()`` needs to be changed to ``ìf done or trunc: env.reset()``.
 - The reset function has also gained an extra return value. This needs to be caught
   to avoid potentially obscure unpacking errors. ``obs = env.reset()`` ->
   ``obs, _ = env.reset()`` will do.
 - Rendering has changed significantly, depending on use case. If you wish to
   use the interactive window you can pass 'render_mode="human"' to the constructor
   and then call ``env.render()``.
   If you want to render images from multiple different cameras, or use both an
   interactive window and also render arrays (for example to save as a video), you
   should use gymnasiums MuJoCoRenderer with
   ``img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name=...)``,
   similar to the old interface.

There are other minor changes as well, please see
