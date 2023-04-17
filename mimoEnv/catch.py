""" Training script for the catch scenario.

This script allows simple training and testing of RL algorithms in the Reach environment with a command line interface.
A selection of RL algorithms from the Stable Baselines3 library can be selected.
Interactive rendering is disabled during training to speed up computation, but enabled during testing, so the behaviour
of the model can be observed directly.

Trained models are saved automatically into the `models` directory and prefixed with `reach`, i.e. if you name your
model `my_model`, it will be saved as `models/reach_my_model`.

To train a given algorithm for some number of timesteps::

    python reach.py --train_for=200000 --test_for=1000 --algorithm=PPO --save_model=<model_suffix>

To review a trained model::

    python reach.py --test_for=1000 --load_model=<your_model_suffix>

The available algorithms are `PPO`, `SAC`, `TD3`, `DDPG` and `A2C`.
"""

import gym
import time
import mimoEnv
from mimoActuation.actuation import TorqueMotorModel
from mimoActuation.muscle import MuscleModel
import argparse
import cv2


def test(env, test_for=1000, model=None, render_video=False):
    """ Testing function to view the behaviour of a model.

    Args:
        env (gym.Env): The environment on which the model should be tested. This does not have to be the same training
            environment, but action and observation spaces must match.
        test_for (int): The number of timesteps the testing runs in total. This will be broken into multiple episodes
            if necessary.
        model:  The stable baselines model object. If ``None`` we take random actions instead.
    """
    env.seed(42)
    obs = env.reset()
    images = []
    im_counter = 0

    for idx in range(test_for):
        if model is None:
            print("No model, taking random actions")
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        #env.render()
        if render_video:
            img = env.render(mode="rgb_array")[0]
            images.append(img)
        if done:
            time.sleep(1)
            obs = env.reset()
            if render_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter('catch_{}.avi'.format(im_counter), fourcc, 50, (500, 500))
                for img in images:
                    video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.destroyAllWindows()
                video.release()

                images = []
                im_counter += 1

    env.reset()


def main():
    """ CLI for this scenario.

    Command line interface that can train and load models for the reach scenario. Possible parameters are:

    - ``--train_for``: The number of time steps to train. No training takes place if this is 0. Default 0.
    - ``--test_for``: The number of time steps to test. Testing renders the environment to an interactive window, so
      the trained behaviour can be observed. Default 1000.
    - ``--save_every``: The number of time steps between model saves. This can be larger than the total training time,
      in which case we save once when training completes. Default 100000.
    - ``--algorithm``: The algorithm to train. This argument must be provided if you train. Must be one of
      ``PPO, SAC, TD3, DDPG, A2C, HER``.
    - ``--load_model``: The model to load. Note that this only takes suffixes, i.e. an input of `my_model` tries to
      load `models/reach_my_model`.
    - ``--save_model``: The name under which we save. Like above this is a suffix.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_for', default=0, type=int,
                        help='Total timesteps of training')
    parser.add_argument('--test_for', default=1000, type=int,
                        help='Total timesteps of testing of trained policy')               
    parser.add_argument('--save_every', default=100000, type=int,
                        help='Number of timesteps between model saves')
    parser.add_argument('--algorithm', default=None, type=str, required=True,
                        choices=['PPO', 'SAC', 'TD3', 'DDPG', 'A2C', 'HER'],
                        help='RL algorithm from Stable Baselines3')
    parser.add_argument('--load_model', default=False, type=str,
                        help='Name of model to load')
    parser.add_argument('--save_model', default='', type=str,
                        help='Name of model to save')
    parser.add_argument('--action_penalty', action='store_true',
                        help='Adds a penalty for actions if true.')
    parser.add_argument('--render_video', action='store_true',
                        help='Renders a video for each episode during the test run.')
    parser.add_argument('--use_muscle', action='store_true',
                        help='Use the muscle actuation model instead of spring-damper model if provided.')
    
    args = parser.parse_args()
    algorithm = args.algorithm
    load_model = args.load_model
    save_model = args.save_model
    save_every = args.save_every
    train_for = args.train_for
    test_for = args.test_for
    action_penalty = args.action_penalty
    render_video = args.render_video
    use_muscle = args.use_muscle
    actuation_model = MuscleModel if use_muscle else TorqueMotorModel

    if algorithm == 'PPO':
        from stable_baselines3 import PPO as RL
    elif algorithm == 'SAC':
        from stable_baselines3 import SAC as RL
    elif algorithm == 'TD3':
        from stable_baselines3 import TD3 as RL
    elif algorithm == 'DDPG':
        from stable_baselines3 import DDPG as RL
    elif algorithm == 'A2C':
        from stable_baselines3 import A2C as RL

    env = gym.make('MIMoCatch-v0', action_penalty=action_penalty, actuation_model=actuation_model)
    _ = env.reset()

    # load pretrained model or create new one
    if algorithm is None:
        model = None
    elif load_model:
        model = RL.load(load_model, env)
    else:
        model = RL("MultiInputPolicy", env, tensorboard_log="models/tensorboard_logs/" + save_model, verbose=1, learning_rate=5e-5)

    # train model
    counter = 0
    while train_for > 0:
        counter += 1
        train_for_iter = min(train_for, save_every)
        train_for = train_for - train_for_iter
        model = model.learn(total_timesteps=train_for_iter, reset_num_timesteps=False)
        model.save("models/catch/" + save_model + "/model_" + str(counter))
    
    test(env, model=model, test_for=test_for, render_video=render_video)


if __name__ == '__main__':
    main()
