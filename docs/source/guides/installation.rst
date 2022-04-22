Installation
============

Installation is very straight forward::

    Install MuJoCo
    Install mujoco-py
    Clone this repository
    pip install -r requirements.txt
    pip install -e .

To test that everything installed correctly run ``python mimoEnv/showroom.py``. A new window
should pop up with a simple room with some toys and MIMo. MIMo should flop to the ground. The
window will close on its own after about half a minute.

All the sample environments register with Gym and can be created using
``gym.make("environment-name")``, but you must ``import mimoEnv`` first.