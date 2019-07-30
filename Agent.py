import multiprocessing
from baselines.common.atari_wrappers import wrap_deepmind,make_atari
import numpy as np
import gym
import collections
import json
import gc
with open('setting.json') as json_file:  
        SETTING = json.load(json_file)
ENV = SETTING['ENV']
NUM_STATE = SETTING['N_STATE']
NUM_ACTIONS = SETTING['N_ACTIONS']
NUM_LSTM = SETTING['N_LSTM']
N_STEP = SETTING['N_STEP_UNROLL']

class Agent(multiprocessing.Process):
        stop_signal = False
        def __init__(self, num=None,render=False, train_buffer = None, model = None, frame=None, seed = 0, pipe = None, push_state = None, list_ep = None, list_reward = None):
                multiprocessing.Process.__init__(self)
                self.num = num
                self.name = 'process_{}'.format(self.num)
                self.env = wrap_deepmind(gym.make(ENV),episode_life=True, clip_rewards=False, frame_stack=True, scale=True)
                self.env.max_episode_steps = None
                self.memory = collections.deque(maxlen=N_STEP)
                self.share_train = train_buffer
                self.frame = frame
                self.step = 0.0          
                self.seed = seed
                self.mean_episode = collections.deque(maxlen=10)
                self.mean_reward = collections.deque(maxlen=10)
                self.pipe = pipe
                self.push_state = push_state
                self.ep_list = list_ep
                self.reward_list = list_reward

        def runEpisode(self):
                s = self.env.reset()
                R=0
                self.step = 0.0

                while 1:
                        p = self.predict(s)
                        a = np.random.choice(NUM_ACTIONS, p=p)
                        s_, r, done, info = self.env.step(a)
                        R+=r
                        a_cats = np.eye(NUM_ACTIONS)[a]
                        if done:
                                self.train(s,a_cats, r, p, s_,0.0)
                        else:   
                                self.train(s,a_cats, r, p, s_,1.0)
                        self.step+=1.0
                        s = s_

                        if done or self.stop_signal:
                                break

                self.frame.value += self.step
                self.current_episode+=1
                self.mean_reward.append(R)
                self.mean_episode.append(self.step)
                self.ep_list[self.num] = self.step
                self.reward_list[self.num] = R
                print("R: %3d"%(int(R)),"step: ",self.frame.value,"episode_step: ",self.step)
                summary = self.session.run(self.write_op,
                                        {
                                                self.report_reward: R,
                                                self.report_episode: self.step,
                                                self.report_reward_mean: np.mean(self.mean_reward),
                                                self.report_episode_mean: np.mean(self.mean_episode),
                                        })
                self.writer.add_summary(summary,self.current_episode)
                self.writer.flush()

        def run(self):
                self.env.seed(self.seed)
                self.current_episode = 0.0
                self.sub_init()
                self.stop_signal=False
                while not self.stop_signal:
                        self.runEpisode()

        def sub_init(self):
                import os, warnings
                os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
                import tensorflow as tf
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                from tensorflow.python.util import deprecation
                deprecation._PRINT_DEPRECATION_WARNINGS = False
                warnings.simplefilter(action='ignore', category=FutureWarning)
                warnings.simplefilter(action='ignore', category=Warning)
                self.count=0.0
                self.n1 = None
                self.n2 = None
                self.writer = tf.summary.FileWriter("./logs/"+ENV+"/"+self.name)
                self.report_reward = tf.Variable(0.0)
                self.report_episode = tf.Variable(0.0)
                self.report_reward_mean = tf.Variable(0.0)
                self.report_episode_mean = tf.Variable(0.0)
                tf.summary.scalar("episode", self.report_episode,family='1. monitor')
                tf.summary.scalar("reward", self.report_reward,family='1. monitor')
                tf.summary.scalar("mean_episode", self.report_episode_mean,family='1. monitor')
                tf.summary.scalar("mean_reward", self.report_reward_mean,family='1. monitor')
                self.write_op = tf.summary.merge_all()
                self.session = tf.Session()
                self.session.run(tf.global_variables_initializer())
                self.soft_reset=False
              

        def stop(self):
                self.stop_signal = True

        def predict(self, s):
                self.push_state.put((s,self.num))
                p_,_n1,_n2,stop_signal= self.pipe.recv()
                self.stop_signal=stop_signal

                if self.n1 is None:
                        self.n1 = _n1
                        self.n2 = _n2
                return p_

        def train(self, s, a_cat, r, p,s_, flag_done):
                self.memory.append((s, a_cat, r, p,s_,flag_done))
                if len(self.memory)>=N_STEP:
                        _s,_a, _r, _p,_s_,_s_mask = zip(*self.memory)
                        s = np.array(_s).astype(np.float32).reshape((-1,*NUM_STATE))
                        a = np.array(_a).astype(np.float32).reshape((-1,NUM_ACTIONS))
                        r = np.array(_r).astype(np.float32).reshape((-1,1))
                        p = np.array(_p).astype(np.float32).reshape((-1,NUM_ACTIONS))
                        mu = np.sum(p*a,axis=-1,keepdims=True).astype(np.float32)
                        s_ = np.array(_s_).astype(np.float32).reshape((-1,*NUM_STATE))
                        s_mask = np.array(_s_mask).astype(np.float32).reshape((-1,1))

                        ready = (s, a, r, mu,s_[-1],s_mask,self.n1,self.n2)
                        self.share_train.put(ready)
                        self.memory.clear()
                        self.n1 = None
                        self.n2 = None

                        gc.collect()