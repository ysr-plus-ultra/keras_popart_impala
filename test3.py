#import time
import gym
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
#import random
from gym.utils import play
from baselines.common.atari_wrappers import wrap_deepmind
from gym import wrappers
import time
from keras.models import load_model
from custom_layers.popart_layer import PopArt
from custom_layers.pixel_shift import Pixel_Shift
from custom_layers.NoisyDense import NoisyDense
from custom_layers.layer_ell_norm import Layer_Ell
model = load_model('./logs/PongDeterministic-v4/PongDeterministic-v4.h5', custom_objects={'Pixel_Shift':Pixel_Shift,'PopArt':PopArt,'NoisyDense':NoisyDense,'Layer_Ell':Layer_Ell })
env = wrap_deepmind(gym.make("PongDeterministic-v4"),episode_life=True, clip_rewards=False, frame_stack=True, scale=True)
direc = './png/' + str(time.time())
import imageio
import numpy as np
images = []

try:
    os.mkdir(direc)
except OSError:
    print ("Creation of the directory %s failed" % direc)
else:
    print ("Successfully created the directory %s " % direc)
s = np.array(env.reset())
s=s.reshape((1,84,84,4))
noise1 = np.zeros((1,256, 4))
noise2 = np.zeros((1,4))
noise3 = np.zeros((1,256, 1))
noise4 = np.zeros((1,1))
i=0
done=False
while 1:
    filename = direc+'/'+'test_image_{}.png'.format(i)
    p,v = model.predict([s,noise1,noise2,noise3,noise4])
    a = np.random.choice(4, p=p[0])
    s_,_,done,_=env.step(a)
    env.render()
    env.env.unwrapped.ale.saveScreenPNG(filename)
    images.append(imageio.imread(filename))
    s_ = np.array(s_)
    s = s_.reshape((1,84,84,4))
    i+=1
    if done:
        break

imageio.mimsave(direc+'/'+'movie.gif', images)
