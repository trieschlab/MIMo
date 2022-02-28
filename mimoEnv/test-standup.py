import gym
import time
import mimoEnv
import argparse
from stable_baselines3 import PPO

def test(env, test_for=1000, model=None):
    obs = env.reset()
    for _ in range(test_for):
        if model == None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.reset()
    env.close()

def main():

    env = gym.make('MIMoStandup-v0')
    env.reset()

    print(env.sim.data.qpos.shape)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_for', default=0, type=int,
                        help='Total timesteps of training')
    parser.add_argument('--test_for', default=1000, type=int,
                        help='Total timesteps of testing of trained policy')               
    parser.add_argument('--save_every', default=100000, type=int,
                        help='Number of timesteps between model saves')
    parser.add_argument('--load_model', default=False, type=str,
                        help='Name of model to load')
    parser.add_argument('--save_model', default='', type=str,
                        help='Name of model to save')
    parser.add_argument('--no_model', default=False, type=bool,
                        help='If True creates environment without a model')
    args = parser.parse_args()
    load_model = args.load_model
    save_model = args.save_model
    no_model = args.no_model
    save_every = args.save_every
    train_for = args.train_for
    test_for = args.test_for

    # load pretrained model or create new one
    if no_model:
        model = None
    elif load_model:
        model = PPO.load("models/standup" + load_model, env)
    else:
        model = PPO("MultiInputPolicy", env, verbose=1)

    # train model
    while train_for>0:
        train_for_iter = min(train_for, save_every)
        train_for = train_for - train_for_iter
        model.learn(total_timesteps=train_for_iter)
        model.save("models/standup" + save_model)
    
    test(env, model=model, test_for=test_for)


if __name__=='__main__':
    main()
