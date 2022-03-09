import gym
import time
import numpy as np
import mimoEnv
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import minmax_scale

def main():
    env = gym.make("MIMoGrasp-v0")
    obs = env.reset()

    grasp_pose = np.array([
        0, 0, 0.35, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        -0.73606, 0.43868, 0, -0.133518, 0, -0.133518, 0, 1.14183, 0.7941, -1.35139, -0.650606, -1.571, -0.23892, -0.078856, -0.872147,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, -0.15, 0.6, 1, 0, 0, 0
    ])
    env.sim.data.qpos[:] = grasp_pose
    obs,_,_,_ = env.step(np.zeros(env.action_space.shape))
    #env.render()
    #time.sleep(1)
    env.close()

    # BINOCULAR VISION
    # Creates two grayscale images from the left and right eyes,
    # then stacks them in color dimension as a 3d stereoscopic image

    img_left = obs['eye_left']
    img_left = Image.fromarray(img_left)
    img_left = img_left.convert('L')
    imgs_min = np.min(img_left)
    imgs_max = np.max(img_left)
    img_std = (img_left-imgs_min) / (imgs_max-imgs_min)
    img_left = img_std*2 - 1

    img_right = obs['eye_right']
    img_right = Image.fromarray(img_right)
    img_right = img_right.convert('L')
    imgs_min = np.min(img_right)
    imgs_max = np.max(img_right)
    img_std = (img_right-imgs_min) / (imgs_max-imgs_min)
    img_right = img_std*2 - 1

    img_left_3d = np.reshape(img_left, img_left.shape + (1,))
    img_right_3d = np.reshape(img_right, img_right.shape + (1,))
    img_center_3d = (img_left_3d + img_right_3d) / 2.0
    img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
    img_stereo = (img_stereo+1)/2

    plt.imshow(img_left)
    plt.show()
    plt.imshow(img_stereo)
    plt.show()

    # TOUCH
    touch = obs['touch']

if __name__=='__main__':
    main()
