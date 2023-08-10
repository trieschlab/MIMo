Installation
============

Installation is very straight forward::

    Clone this repository
    pip install -r requirements.txt
    pip install -e .

To test that everything installed correctly run ``python mimoEnv/showroom.py``. A new window
should pop up with a simple room with some toys and MIMo. MIMo should flop to the ground. The
window will close on its own after about half a minute.

The other demo scripts additionally use `Stable Baselines3 <https://github.com/DLR-RM/stable-baselines3>`_
algorithms for training.

All the sample environments register with gym and can be created using
``gym.make("environment-name")``, but you must ``import mimoEnv`` first.
