import json
import multiprocessing
from baselines.common.atari_wrappers import *
import numpy as np
import random

with open('setting.json') as json_file:  
        SETTING = json.load(json_file)

NUM_STATE = SETTING['N_STATE']
NUM_ACTIONS = SETTING['N_ACTIONS']
NUM_LSTM = SETTING['N_LSTM']
N_STEP = SETTING['N_STEP_UNROLL']
ENV = SETTING['ENV']
NUM_TASK = len(ENV)
NOISY = SETTING['NOISYNET']

class Agent(multiprocessing.Process):
        stop_signal = False
        def __init__(self, num=None, task=None, render=False, train_buffer = None, model = None, frame=None, seed = 0, pipe = None, push_state = None, list_ep = None, list_reward = None):

                multiprocessing.Process.__init__(self)
                self.num = num
                self.task = task
                self.taskidx = self.num%NUM_TASK
                self.name = 'process_{}'.format(self.num)
                self.memory = []
                self.share_train = train_buffer
                self.frame = frame
                self.step = 0.0
                self.seed = seed
                self.pipe = pipe
                self.push_state = push_state
                self.ep_list = list_ep
                self.reward_list = list_reward

        def runEpisode(self):
                s=self.env.reset()
                while 1:
                        p = self.predict(s)
                        a = np.random.choice(NUM_ACTIONS, p=p)
                        s_, r, done, _ = self.env.step(a)
                        self.R+=r
                        a_cats = np.eye(NUM_ACTIONS)[a]
                        if done:
                                self.train(s, a_cats, r, p, s_,0.0)
                        else:   
                                self.train(s, a_cats, r, p, s_,1.0)
                        self.step+=1.0
                        s = s_

                        if done or self.stop_signal:
                                break

                if self.env.unwrapped.ale.lives()==0:
                        if self.ep_list[self.taskidx] == 0:
                                self.ep_list[self.taskidx] = self.step
                                self.reward_list[self.taskidx] = self.R
                        else:
                                self.ep_list[self.taskidx] = self.ep_list[self.taskidx]*0.99 + 0.01*self.step
                                self.reward_list[self.taskidx] = self.reward_list[self.taskidx]*0.99 + 0.01*self.R

                        self.R = 0
                        self.step=0

        def run(self):
                self.sub_init()
                self.stop_signal=False
                while not self.stop_signal:
                        self.runEpisode()

        def make_env(self):
                env = gym.make(self.task)
                env = MaxAndSkipEnv(env, skip=4)
                env = EpisodicLifeEnv(env)
                env = WarpFrame(env)
                env = FrameStack(env, 4)
                env.seed(self.seed)

                return env

        def sub_init(self):
                self.env=self.make_env()
                self.R = 0
                self.step= 0
                self.reset_noise()

        def stop(self):
                self.stop_signal = True

        def reset_noise(self):
                if NOISY:
                        self.noise1 = np.random.normal(size=(NUM_LSTM,NUM_ACTIONS))
                        self.noise2 = np.random.normal(size=(NUM_ACTIONS))
                else:
                        self.noise1 = np.zeros((NUM_LSTM,NUM_ACTIONS))
                        self.noise2 = np.zeros((NUM_ACTIONS))


        def predict(self, s):
                self.push_state.put((s,self.num,self.noise1,self.noise2))
                p_,stop_signal= self.pipe.recv()
                self.stop_signal=stop_signal

                return p_

        def train(self, s,a_cat, r, p, s_,flag_done):
                self.memory.append((s,a_cat, r, p,s_,flag_done))
                if len(self.memory)>=N_STEP:
                        _s,_a, _r, _p,_s_,_s_mask = zip(*self.memory)
                        s = np.array(_s,dtype=np.uint8)
                        a = np.array(_a,dtype=np.float32)
                        r = np.array(_r,dtype=np.float32).reshape((-1,1))
                        p = np.array(_p,dtype=np.float32)
                        mu = np.sum(p*a,axis=-1,keepdims=True,dtype=np.float32).reshape((-1,1))
                        s_ = np.array(_s_,dtype=np.uint8)
                        s_mask = np.array(_s_mask,dtype=np.uint8).reshape((-1,1))
                        ready = (s, a, r, mu,s_[-1],s_mask,self.taskidx,self.noise1,self.noise2)
                        self.share_train.put_nowait(ready)
                        self.memory.clear()
                        self.reset_noise()





