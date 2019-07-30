import os, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing,threading
from Agent import Agent
import numpy as np
import json
import random

with open('setting.json') as json_file:  
        SETTING = json.load(json_file)

ENV = SETTING['ENV']
NUM_STATE = SETTING['N_STATE']
NUM_ACTIONS = SETTING['N_ACTIONS']
NUM_LSTM = SETTING['N_LSTM']
N_STEP_UNROLL = SETTING['N_STEP_UNROLL']
USE_REPLAY = SETTING['REPLAY']
N_REPLAY = SETTING['REPLAY_BUFFER']
ACTORS = SETTING['ACTOR']
ENTROPY = SETTING['ENTROPY']
NOISYNET = SETTING['NOISYNET']
POPART = SETTING['POPART']
BATCH_SIZE = SETTING['BATCH']
FEEDERS = SETTING['FEEDER']
MAX_TRY = SETTING['MAX_TRY']
GAMMA = SETTING['GAMMA']
LEARNING_RATE = SETTING['LEARNING_RATE']
POPART_BETA =SETTING['POPART_BETA']

if __name__ == '__main__':
        train_manager = multiprocessing.Manager()
        seeds = random.sample(range(1, 1000), ACTORS)
        print("random seed: ",seeds)
        shared_training_buffer = train_manager.Queue(ACTORS)
        shared_step = train_manager.Value('d', 0.0)
        policy_state_push = train_manager.Queue(ACTORS)
        report_episode = train_manager.list([np.nan]*ACTORS)
        report_reward = train_manager.list([np.nan]*ACTORS)
        learner_pipe=[]
        actor_pipe=[]

        for i in range(ACTORS):
                conn1, conn2 = multiprocessing.Pipe()
                learner_pipe.append(conn1)
                actor_pipe.append(conn2)
        envs = [Agent(num=i,
                        train_buffer = shared_training_buffer,
                        frame = shared_step, seed=seeds[i], 
                        pipe = actor_pipe[i], 
                        push_state=policy_state_push, 
                        list_ep = report_episode, 
                        list_reward = report_reward) for i in range(ACTORS)]
        #start sub-process
        for e in envs:
                e.daemon=True
                e.start()

        import collections
        import io
        import gc
        import math

        from pprint import pprint
        import time
        import xml.etree.ElementTree as ET

        import tensorflow as tf

        from tensorflow.python import debug as tf_debug
        import queue
        from Model import build_model




        class Brain:
                replay_buffer = collections.deque(maxlen=N_REPLAY)
                train_queue = None
                locking = threading.Lock()
                def __init__(self):
                        import keras.backend.tensorflow_backend as K
                        config = tf.ConfigProto()
                        config.gpu_options.allow_growth = True
                        self.session = tf.Session(config=config)
                        K.set_session(self.session)
                        K.manual_variable_initialization(True)
                        self.lp = K.learning_phase()
                        
                        self.model = build_model()
                        self.model._make_predict_function()
                        self.model.summary()

                        self.predict_model = build_model()
                        self.predict_model._make_predict_function()
        
                        self.graph = self._build_graph(self.model)
                        self.popart = self.popart_ops()
                        self.session.run(tf.global_variables_initializer())
                        self.default_graph = tf.get_default_graph()
                        self.sync_weight()
                        self.tensorboard_setting()
                        self.default_graph.finalize()
                        self.qq = queue.Queue(maxsize=16)
                        
                def tensorboard_setting(self):
                        tf.summary.scalar('loss_policy',tf.reduce_mean(self.loss_policy),family='2. loss')
                        tf.summary.scalar('loss_value',tf.reduce_mean(self.loss_value),family='2. loss')
                        tf.summary.scalar('loss_entropy',tf.reduce_mean(self.entropy),family='2. loss')
                        tf.summary.scalar('kernel_weight',tf.reduce_mean(tf.abs(self.pop_art_layer.kernel)),family='6.noise_monitor')
                        tf.summary.scalar('kernel_noise',tf.reduce_mean(tf.abs(self.pop_art_layer.sigmaW)),family='6.noise_monitor')
                        
                        self.process_reward = tf.Variable(0.0)
                        self.process_ep = tf.Variable(0.0)

                        tf.summary.scalar("thread_episode_by_time", self.process_ep,family='0. threads')
                        tf.summary.scalar("thread_reward_by_time", self.process_reward,family='0. threads')
                        self.merged = tf.summary.merge_all()
                        self.train_writer = tf.summary.FileWriter("./logs/"+ENV)

                def popart_ops(self):
                        popart_mu = self.pop_art_layer.mu
                        popart_nu = self.pop_art_layer.nu
                        popart_kernel = self.pop_art_layer.kernel
                        popart_bias = self.pop_art_layer.bias

                        new_kernel = popart_kernel*self.old_sigma/self.updated_sigma
                        new_bias = (self.old_sigma*popart_bias+self.old_mu-self.updated_mu)/self.updated_sigma

                        assign_mu = tf.assign(popart_mu,self.updated_mu)
                        assign_nu = tf.assign(popart_nu,self.updated_nu)
                        assign_kernel = tf.assign(popart_kernel,new_kernel)
                        assign_bias = tf.assign(popart_bias,new_bias)

                        return (assign_mu,assign_nu,assign_kernel,assign_bias)

                        
                def _build_graph(self,model):

                        s_t = tf.placeholder(tf.float32, shape=(N_STEP_UNROLL,None,*NUM_STATE),name='s_t')
                        a_t = tf.placeholder(tf.float32, shape=(N_STEP_UNROLL, None, NUM_ACTIONS),name='a_t')
                        r_t = tf.placeholder(tf.float32, shape=(N_STEP_UNROLL, None, 1),name='r_t')
                        mu_t = tf.placeholder(tf.float32, shape=(N_STEP_UNROLL, None, 1),name='mu_t')
                        _s_t = tf.placeholder(tf.float32, shape=(None, *NUM_STATE),name='s_next_t')
                        _s_mask_t = tf.placeholder(tf.float32, shape=(N_STEP_UNROLL,None,1),name='s_mast_t')

                        noise1_t = tf.placeholder(tf.float32, shape=(None, NUM_LSTM, NUM_ACTIONS),name='noise1_t')
                        noise2_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS),name='noise2_t')
                        noise3_t = tf.placeholder(tf.float32, shape=(None, NUM_LSTM, 1),name='noise3_t')
                        noise4_t = tf.placeholder(tf.float32, shape=(None, 1),name='noise4_t')

                        placeholder_set = s_t, a_t, r_t, mu_t, _s_t, _s_mask_t,noise1_t,noise2_t,noise3_t,noise4_t
                        discount_t = _s_mask_t*GAMMA

                        d_size = tf.cast(tf.size(r_t),tf.float32)

                        self.pop_art_layer = None
                        self.preprocessing = None
                        for elem in model.layers:
                                if "pop" in elem.name:
                                        self.pop_art_layer=elem
                                elif "pixel" in elem.name:
                                        self.preprocessing = elem
                                        
                        self.old_mu = tf.Variable(0.0,trainable=False)
                        self.old_nu = tf.Variable(1.0,trainable=False)
                        self.old_sigma = tf.Variable(1.0,trainable=False)

                        old_mu          = tf.assign(self.old_mu,self.pop_art_layer.mu)
                        old_nu          = tf.assign(self.old_nu,self.pop_art_layer.nu)
                        old_sigma       = tf.clip_by_value(tf.sqrt(old_nu-tf.square(old_mu)),1e-4,1e+6)
                        old_sigma       = tf.assign(self.old_sigma,old_sigma)

                        self.updated_mu = tf.Variable(0.0,trainable=False)
                        self.updated_nu = tf.Variable(1.0,trainable=False) 
                        self.updated_sigma = tf.Variable(1.0,trainable=False)               

                        sequence_items = (s_t)
                        initial_values = (tf.zeros_like(a_t[0]),tf.zeros_like(r_t[0]))
                        def calc_hc(last_output, current_sequence):
                                current_s= current_sequence
                                p, v = model([current_s, noise1_t,noise2_t,noise3_t,noise4_t])
                                return p,v

                        p,normalized_v = tf.scan(fn=calc_hc,
                                                        elems=sequence_items,
                                                        initializer=initial_values,
                                                        parallel_iterations=1,
                                                        back_prop=True,
                                                        name='scan1')

                        _,bootstrap_value = model([_s_t,noise1_t,noise2_t,noise3_t,noise4_t])

                        unnormalized_v = normalized_v*old_sigma+old_mu
                        bootstrap_value = bootstrap_value*old_sigma+old_mu
                        
                        v_plus1 = tf.concat([unnormalized_v[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
                        pi = tf.reduce_sum(p*a_t,axis=-1,keepdims=True)
                        rho_s = tf.where(tf.equal(mu_t,0.),tf.zeros_like(mu_t),pi/mu_t)
                        rho_thres = 1.0
                        rho_s = tf.minimum(rho_s,rho_thres) 
                        c_s = tf.minimum(rho_s,1.0)                                                                
                        deltas = c_s*(r_t+discount_t*v_plus1-unnormalized_v)
                        sequences = (c_s,deltas,discount_t)

                        def scanfunc(acc, sequence_item):
                                c_seq, delta_seq, discount_seq = sequence_item
                                return delta_seq + discount_seq * c_seq * acc                                 

                        initial_values = tf.zeros_like(bootstrap_value)
                        vs_minus_v_xs = tf.scan(
                                        fn=scanfunc,
                                        elems=sequences,
                                        initializer=initial_values,
                                        parallel_iterations=1,
                                        back_prop=False,
                                        reverse=True,  # Computation starts from the back.
                                        name='scan2')
                        
                        G_v = tf.stop_gradient(unnormalized_v+vs_minus_v_xs) 

                        self.update_statistics(G_v, old_mu, old_nu)

                        normalized_G_v = tf.stop_gradient((G_v-old_mu)/old_sigma)

                        advantage = normalized_G_v-normalized_v

                        safe_p = tf.where(tf.equal(p, 0.), tf.ones_like(p), p)
                        log_prob =  a_t*tf.log(safe_p)
                        
                        self.loss_policy = -log_prob * tf.stop_gradient(advantage)                             # maximize policy
                        self.loss_value  = tf.square(advantage)                                                  # minimize value error

                        self.entropy = tf.reduce_sum(p * tf.log(safe_p), axis=-1, keepdims=True)    # maximize entropy (regularization)

                        tf.summary.scalar('max_prob',tf.reduce_mean(tf.reduce_max(p,axis=-1,keepdims=True)),family='5. etc')
                        tf.summary.scalar('mean_value',tf.reduce_mean(unnormalized_v),family='5. etc')
                        tf.summary.scalar('old_mu',self.old_mu,family='6. stat')
                        tf.summary.scalar('old_nu',self.old_nu,family='6. stat')
                        tf.summary.scalar('old_sigma',self.old_sigma,family='6. stat')

                        self.loss_total = tf.reduce_sum(self.loss_policy + 0.5 *self.loss_value +  ENTROPY * self.entropy)
                        self.g_step = tf.train.get_or_create_global_step()
                        
                        LR=LEARNING_RATE
                        m = tf.reduce_mean(s_t[...,0],axis=(0,1))
                        mean_update = tf.assign(self.preprocessing.mean,self.preprocessing.mean*0.997+(1-0.997)*m)
                        self.update_ops = [mean_update]                                 
                        optimizer = tf.train.RMSPropOptimizer(LR)
                        
                        #with tf.control_dependencies(self.update_ops):
                        minimize = optimizer.minimize(self.loss_total,global_step=self.g_step)
                
                        return placeholder_set, minimize

                def update_statistics(self, G_v, old_mu, old_nu):
                        self.beta = POPART_BETA
                        new_mu = tf.reduce_mean(G_v)
                        new_nu = tf.reduce_mean(tf.square(G_v))

                        updated_mu = (1-self.beta)*old_mu+self.beta*new_mu
                        updated_nu = (1-self.beta)*old_nu+self.beta*new_nu
                        updated_sigma = tf.clip_by_value(tf.sqrt(updated_nu-tf.square(updated_mu)),1e-4,1e+6)

                        assign_updated_mu = tf.assign(self.updated_mu, updated_mu)
                        assign_updated_nu = tf.assign(self.updated_nu, updated_nu)
                        assign_updated_sigma = tf.assign(self.updated_sigma, updated_sigma)

                        self.update_moving_average = (assign_updated_mu,assign_updated_nu,assign_updated_sigma)

                def get_unroll(self):
                        batch_list = []
                        if USE_REPLAY:
                                B = BATCH_SIZE//2
                        else:
                                B = BATCH_SIZE
                        
                        for i in range(B):
                                if NOISYNET:
                                        noise3 = np.random.normal(size=(NUM_LSTM, 1))
                                        noise4 = np.random.normal(size=(1))
                                else:
                                        noise3 = np.zeros((NUM_LSTM, 1))
                                        noise4 = np.zeros((1))
                                elem = self.train_queue.get()
                                batch_list.append(elem+(noise3,noise4))
                        
                        if USE_REPLAY:
                                self.replay_buffer.extend(batch_list)
                                if len(self.replay_buffer)>B:
                                        replay = random.sample(self.replay_buffer,B)                       
                                        batch_list = batch_list+replay
                        return batch_list
                                
                def feed(self):
                        trainq = self.get_unroll()
                        s,a,r,mu,s_,s_mask,noise1,noise2,noise3,noise4=([],[],[],[],[],[],[],[],[],[])
                        for elem in trainq:
                                _s, _a, _r, _mu,_s_,_s_mask,_n1,_n2,_n3,_n4 = elem
                                s.append(_s)
                                a.append(_a)
                                r.append(_r)
                                mu.append(_mu)
                                s_.append(_s_)
                                s_mask.append(_s_mask)
                                noise1.append(_n1)
                                noise2.append(_n2)
                                noise3.append(_n3)
                                noise4.append(_n4)

                        s = np.moveaxis(np.array(s),0,1)

                        a = np.moveaxis(np.array(a),0,1)
                        r = np.moveaxis(np.array(r),0,1)
                        mu = np.moveaxis(np.array(mu),0,1)
                        s_ = np.array(s_)

                        s_mask = np.moveaxis(np.array(s_mask),0,1)
                        noise1 = np.array(noise1)
                        noise2 = np.array(noise2)
                        noise3 = np.array(noise3)
                        noise4 = np.array(noise4)
                        self.qq.put((s,a,r,mu,s_,s_mask,noise1,noise2,noise3,noise4))

                def optimize(self):
                        s,a,r,mu,s_,s_mask,noise1,noise2,noise3,noise4 = self.qq.get()
                        placeholder_set, minimize = self.graph
                        s_t, a_t, r_t, mu_t, _s_t, _s_mask_t,noise1_t,noise2_t,noise3_t,noise4_t = placeholder_set

                        _, summary,TRY,_,_= self.session.run(\
                                        [minimize,self.merged,self.g_step,self.update_moving_average, self.update_ops],\
                                        feed_dict={s_t  : s,
                                                a_t  : a,
                                                r_t  : r,
                                                mu_t : mu,
                                                _s_t: s_,
                                                _s_mask_t: s_mask,
                                                noise1_t: noise1,
                                                noise2_t: noise2,
                                                noise3_t: noise3,
                                                noise4_t: noise4,
                                                self.process_reward:np.nanmean(report_reward),
                                                self.process_ep:np.nanmean(report_episode),
                                                self.lp: 1.0})
                        if POPART:
                                self.session.run(self.popart)
                        
                        if TRY%10==1:
                                self.train_writer.add_summary(summary,global_step=TRY)
                                self.train_writer.flush()
                        if TRY%100==1:
                                print(TRY)
                                self.model.save("./logs/"+ENV+"/"+ENV+".h5")
                        gc.collect()

                def predict(self,noise1,noise2, stop):
                        s,num,n1,n2=[[],[],[],[]]
                        for i in range(ACTORS):
                                _s,_num = policy_state_push.get()
                                s.append(_s)
                                num.append(_num)
                                n1.append(noise1[_num])
                                n2.append(noise2[_num])

                        s = np.array(s)
                        n1 = np.array(n1)
                        n2 = np.array(n2)

                        with self.default_graph.as_default():
                                r_p, _ = self.predict_model.predict([s,n1,n2,np.zeros((BATCH_SIZE, NUM_LSTM, 1)),np.zeros((BATCH_SIZE, 1))])

                        for i in range(ACTORS):
                                idx = num[i]
                                learner_pipe[idx].send((r_p[i], n1[i],n2[i],stop))

                def sync_weight(self):
                        with self.default_graph.as_default():
                                self.predict_model.set_weights(self.model.get_weights())

                

        #---------
        class Optimizer(threading.Thread):
                stop_signal = False

                def __init__(self):
                        threading.Thread.__init__(self)

                def run(self):
                        while not self.stop_signal:
                                brain.optimize()

                def stop(self):
                        self.stop_signal = True

        class Feeder(threading.Thread):
                stop_signal = False

                def __init__(self):
                        threading.Thread.__init__(self)

                def run(self):
                        while not self.stop_signal:
                                brain.feed()

                def stop(self):
                        self.stop_signal = True


        class Batcher(threading.Thread):
                stop_signal = False

                def __init__(self):
                        threading.Thread.__init__(self)

                def run(self):
                        while not self.stop_signal:
                                brain.sync_weight()
                                if NOISYNET:
                                        noise1 = np.random.normal(size=(ACTORS, NUM_LSTM, NUM_ACTIONS))
                                        noise2 = np.random.normal(size=(ACTORS, NUM_ACTIONS))
                                else:
                                        noise1 = np.zeros((ACTORS, NUM_LSTM, NUM_ACTIONS))
                                        noise2 = np.zeros((ACTORS, NUM_ACTIONS))
                                for i in range(N_STEP_UNROLL):
                                        brain.predict(noise1,noise2,self.stop_signal)
                                if self.stop_signal:
                                        break

                def stop(self):
                        self.stop_signal = True

        #-- main
        feeders = [Feeder() for i in range(FEEDERS)]
        opts = Optimizer()
        bats = Batcher()
        feeders.extend([opts,bats])
        brain = Brain()
        brain.train_queue = shared_training_buffer

        print("Learning start")
        start_time = time.time()
        for f in feeders:
                f.daemon=True
                f.start()

        while 1:
                time.sleep(1)
                elapsed_time = time.time()-start_time
                if elapsed_time>3600*1:
                        for f in feeders:
                                f.stop()

                        for e in envs:
                                e.terminate()

                        break