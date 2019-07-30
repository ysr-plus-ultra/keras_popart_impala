#import time
import gym
import os
#import random
from gym.utils import play
from baselines.common.atari_wrappers import wrap_deepmind
from gym import wrappers
import time
env = gym.make('PongDeterministic-v4')
print(env.unwrapped.frameskip)
print(env.unwrapped.ale.getFloat('repeat_action_probability'))
'''
direc = './png/' + str(time.time())
import imageio
images = []


try:
    os.mkdir(direc)
except OSError:
    print ("Creation of the directory %s failed" % direc)
else:
    print ("Successfully created the directory %s " % direc)
env.reset()
i=0
while 1:
    for action in (0,0,0,1,1,1,2,2,2,3,3,3,):
        filename = direc+'/'+'test_image_{}.png'.format(i)
        _,_,done,_=env.step(action)
        env.env.ale.saveScreenPNG(filename)
        images.append(imageio.imread(filename))
        i+=1
        if done:
            break
    if done:
        break

imageio.mimsave(direc+'/'+'movie.gif', images)
'''