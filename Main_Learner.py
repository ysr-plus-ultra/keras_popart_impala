import os, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing,threading
from Agent import Agent
import numpy as np
import json
import random
from itertools import cycle

with open('setting.json') as json_file:  
        SETTING = json.load(json_file)

ENV = SETTING['ENV']
NUM_STATE = SETTING['N_STATE']
NUM_ACTIONS = SETTING['N_ACTIONS']
NUM_LSTM = SETTING['N_LSTM']
N_STEP_UNROLL = SETTING['N_STEP_UNROLL']
USE_REPLAY = SETTING['REPLAY']
N_REPLAY = SETTING['REPLAY_BUFFER']
N_SAMPLE = SETTING['REPLAY_SAMPLE']
ACTORS = SETTING['ACTOR']
ENTROPY = SETTING['ENTROPY']
POPART = SETTING['POPART']
BATCH_SIZE = SETTING['BATCH']
MAX_TRY = SETTING['MAX_TRY']
GAMMA = SETTING['GAMMA']
LEARNING_RATE = SETTING['LEARNING_RATE']
POPART_BETA =SETTING['POPART_BETA']
NUM_TASK = SETTING['N_TASK']

if __name__ == '__main__':
        seeds = random.sample(range(1, 1000), ACTORS)
        man = multiprocessing.Manager()
        print("random seed: ",seeds)
        shared_training_buffer = multiprocessing.Queue(1024)
        shared_step = multiprocessing.Value('d', 0.0)
        policy_state_push = multiprocessing.Queue(ACTORS*2)
        report_episode = man.list([0.0]*NUM_TASK)
        report_reward = man.list([0.0]*NUM_TASK)
        learner_pipe=[]
        actor_pipe=[]
        tasks = cycle(ENV)
        for i in range(ACTORS):
                conn1, conn2 = multiprocessing.Pipe()
                learner_pipe.append(conn1)
                actor_pipe.append(conn2)
        envs = [Agent(num=i,
                        task= next(tasks),
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

        import math

        from pprint import pprint
        import time
        import xml.etree.ElementTree as ET

        import tensorflow as tf
        from tensorflow.python.training import moving_averages
        import queue
        from Model import build_model

        class Brain:
                replay_buffer = collections.deque(maxlen=N_REPLAY)
                train_queue = None
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
                        self.taskqueue = queue.Queue(maxsize=16)
                        
                def tensorboard_setting(self):
                        tf.summary.scalar('loss_policy',tf.reduce_mean(self.loss_policy),family='2. loss')
                        tf.summary.scalar('loss_value',tf.reduce_mean(self.loss_value),family='2. loss')
                        tf.summary.scalar('loss_entropy',tf.reduce_mean(self.entropy),family='2. loss')

                        self.var_ep = tf.Variable(np.zeros(NUM_TASK),dtype=tf.float32)
                        self.var_reward = tf.Variable(np.zeros(NUM_TASK),dtype=tf.float32)
                        tf.summary.scalar('pop_art_kernel',tf.reduce_mean(tf.abs(self.pop_art_layer.kernel)),family='etc')
                        for i in range(NUM_TASK):
                                tf.summary.scalar('episode',self.var_ep[i],family=ENV[i])
                                tf.summary.scalar('reward',self.var_reward[i],family=ENV[i])
                                tf.summary.scalar('mu',self.pop_art_layer.mu[i],family=ENV[i])
                                tf.summary.scalar('sigma',self.pop_art_layer.sigma[i],family=ENV[i])
                        self.merged = tf.summary.merge_all()
                        self.train_writer = tf.summary.FileWriter("./logs/"+"atari")

                def popart_ops(self):
                        popart_mu = self.old_mu
                        popart_sigma = self.old_sigma

                        popart_kernel = self.pop_art_layer.kernel
                        popart_bias = self.pop_art_layer.bias

                        new_kernel = popart_kernel*popart_sigma/self.updated_sigma
                        new_bias = ((popart_sigma*popart_bias)+popart_mu-self.updated_mu)/self.updated_sigma

                        assign_mu = tf.assign(self.pop_art_layer.mu,self.updated_mu)
                        assign_sigma = tf.assign(self.pop_art_layer.sigma,self.updated_sigma)
                        assign_kernel = tf.assign(popart_kernel,new_kernel)
                        assign_bias = tf.assign(popart_bias,new_bias)

                        return (assign_mu,assign_sigma,assign_kernel,assign_bias)

                def update_statistics(self, G_v, task):
                        self.beta = 1-POPART_BETA
                        gv=G_v*task
                        num_data = tf.reduce_sum(task,axis=(0,1))
                        new_mu = tf.reduce_sum(gv,axis=(0,1))/num_data
                        new_nu = tf.reduce_sum(tf.square(gv),axis=(0,1))/num_data
                        new_sigma = tf.sqrt((new_nu-tf.square(new_mu)))
                        new_sigma = tf.clip_by_value(new_sigma,1e-4,1e+6)

                        updated_mu = moving_averages.assign_moving_average(self.updated_mu, new_mu, self.beta)
                        updated_sigma = moving_averages.assign_moving_average(self.updated_sigma, new_sigma, self.beta)

                        self.update_moving_average = (updated_mu,updated_sigma)

                def _build_graph(self,model):

                        s_t = tf.placeholder(tf.float32, shape=(N_STEP_UNROLL,None,*NUM_STATE),name='s_t')
                        a_t = tf.placeholder(tf.float32, shape=(N_STEP_UNROLL, None, NUM_ACTIONS),name='a_t')
                        r_t = tf.placeholder(tf.float32, shape=(N_STEP_UNROLL, None, 1),name='r_t')
                        mu_t = tf.placeholder(tf.float32, shape=(N_STEP_UNROLL, None, 1),name='mu_t')
                        _s_t = tf.placeholder(tf.float32, shape=(None, *NUM_STATE),name='s_next_t')
                        _s_mask_t = tf.placeholder(tf.float32, shape=(N_STEP_UNROLL,None,1),name='s_mast_t')
                        task_t = tf.placeholder(tf.float32, shape=(N_STEP_UNROLL, None, NUM_TASK),name='task_t')

                        placeholder_set = s_t, a_t, r_t, mu_t, _s_t, _s_mask_t, task_t
                        discount_t = _s_mask_t*GAMMA

                        self.pop_art_layer = None
                        self.preprocessing = None
                        for elem in model.layers:
                                if "pop" in elem.name:
                                        self.pop_art_layer=elem
                                elif "pixel" in elem.name:
                                        self.preprocessing = elem
                        
                        self.old_mu = tf.Variable(np.zeros(NUM_TASK),dtype=tf.float32,trainable=False)
                        old_mu = tf.assign(self.old_mu,self.pop_art_layer.mu)
                        self.old_sigma = tf.Variable(np.ones(NUM_TASK),dtype=tf.float32,trainable=False)
                        old_sigma = tf.assign(self.old_sigma,self.pop_art_layer.sigma)

                        self.updated_mu = tf.Variable(np.zeros(NUM_TASK),dtype=tf.float32,trainable=False)
                        self.updated_sigma = tf.Variable(np.ones(NUM_TASK),dtype=tf.float32,trainable=False)                        

                        sequence_items = (s_t)
                        initial_values = (tf.zeros_like(a_t[0]),tf.zeros_like(task_t[0]),tf.zeros_like(task_t[0]))
                        
                        def calc_hc(last_output, current_sequence):
                                current_s= current_sequence
                                p, v, u_v = model([current_s])
                                return p,v,u_v

                        p,normalized_v,unnormalized_v = tf.scan(fn=calc_hc,
                                                        elems=sequence_items,
                                                        initializer=initial_values,
                                                        parallel_iterations=1,
                                                        back_prop=True,
                                                        name='scan1')

                        _,_,bootstrap_value = model([_s_t])

                        v_plus1 = tf.concat([unnormalized_v[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
                        pi = tf.reduce_sum(p*a_t,axis=-1,keepdims=True)
                        rho_s = tf.where(tf.equal(mu_t,0.0),tf.zeros_like(mu_t),tf.truediv(pi,mu_t))
                        rho_thres = 1.0
                        rho_s = tf.minimum(rho_s,rho_thres)                                                               
                        deltas = rho_s*(r_t + discount_t * v_plus1 - unnormalized_v)
                        sequences = (rho_s,deltas,discount_t)

                        def scanfunc(acc, sequence_item):
                                rho_seq, delta_seq, discount_seq = sequence_item
                                return delta_seq + (discount_seq * rho_seq * acc)                                 

                        initial_values = tf.zeros_like(bootstrap_value)
                        vs_minus_v_xs = tf.scan(
                                        fn=scanfunc,
                                        elems=sequences,
                                        initializer=initial_values,
                                        parallel_iterations=1,
                                        back_prop=False,
                                        reverse=True,  # Computation starts from the back.
                                        name='scan2')
                        
                        v_s = tf.stop_gradient(unnormalized_v+vs_minus_v_xs) 
                        G_v = v_s

                        self.update_statistics(G_v, task_t)

                        normalized_G_v = tf.stop_gradient((G_v-old_mu)/old_sigma)

                        advantage = (normalized_G_v-normalized_v)*task_t

                        safe_p = tf.where(tf.equal(p, 0.), tf.ones_like(p), p)
                        log_prob =  tf.reduce_sum(a_t*tf.log(safe_p),axis=-1,keepdims=True)
                        
                        self.loss_policy = -log_prob * tf.stop_gradient(advantage)                                      # maximize policy
                        self.loss_value  = tf.square(advantage)                                                         # minimize value error

                        self.pg_loss = tf.reduce_sum((self.loss_policy + 0.5 *self.loss_value)*task_t,axis=-1,keepdims=True)

                        self.g_step = tf.train.get_or_create_global_step()
                        self.entropy = tf.reduce_sum(p * tf.log(safe_p),axis=-1,keepdims=True)                        # maximize entropy (regularization)
                        
                        self.loss_total = tf.reduce_sum(self.pg_loss + ENTROPY * self.entropy)
                        #self.loss_total = tf.reduce_sum(self.loss_policy + 0.5 *self.loss_value + (ENTROPY/tf.cast(self.g_step+1,tf.float32) * self.entropy))
                        
                        LR=tf.train.polynomial_decay(LEARNING_RATE, self.g_step*N_STEP_UNROLL*(N_SAMPLE+BATCH_SIZE)*4, 2e8, end_learning_rate=1e-4)
                        tf.summary.scalar('learning_rate',LR,family='etc')
                        optimizer = tf.train.RMSPropOptimizer(LR)
                        #optimizer =tf.train.MomentumOptimizer(LR,0.99,use_nesterov=True)
                        #optimizer = tf.train.AdamOptimizer(LR)
                        gradients, variables = zip(*optimizer.compute_gradients(self.loss_total))
                        gradients, _ = tf.clip_by_global_norm(gradients, 40.0)
                        
                        minimize = optimizer.apply_gradients(zip(gradients, variables),global_step=self.g_step)
                        #minimize = optimizer.minimize(self.loss_total,global_step=self.g_step)
                
                        return placeholder_set, minimize

                def get_unroll(self):
                        batch_list = []
                        replay=[]
                        for i in range(BATCH_SIZE):
                                elem = self.train_queue.get()
                                batch_list.append(elem)
                                
                        if len(self.replay_buffer)>N_SAMPLE:
                                replay = random.sample(self.replay_buffer,N_SAMPLE) 
                        if USE_REPLAY:
                                self.replay_buffer.extend(batch_list)

                        return batch_list+replay


                def optimize(self):
                        trainq = self.get_unroll()
                        
                        batch_size = len(trainq)
                        s=np.empty((N_STEP_UNROLL,batch_size,*NUM_STATE),dtype=np.float32)
                        a=np.empty((N_STEP_UNROLL, batch_size, NUM_ACTIONS),dtype=np.float32)
                        r=np.empty((N_STEP_UNROLL, batch_size, 1),dtype=np.float32)
                        mu=np.empty((N_STEP_UNROLL, batch_size, 1),dtype=np.float32)
                        s_=np.empty((batch_size, *NUM_STATE),dtype=np.float32)
                        s_mask=np.empty((N_STEP_UNROLL, batch_size, 1),dtype=np.float32)
                        task=np.empty((N_STEP_UNROLL, batch_size, NUM_TASK),dtype=np.float32)

                        for idx, val in enumerate(trainq):
                                _s, _a, _r, _mu,_s_,_s_mask,_taskidx = val
                                s[:,idx]=_s
                                a[:,idx]=_a
                                r[:,idx]=_r
                                mu[:,idx]=_mu
                                s_[idx]=_s_
                                s_mask[:,idx]=_s_mask
                                task[:,idx]=np.eye(NUM_TASK)[_taskidx]

                        s/=255.0
                        s_/=255.0
                        placeholder_set, minimize = self.graph
                        s_t, a_t, r_t, mu_t, _s_t, _s_mask_t, task_t= placeholder_set

                        _, summary,TRY,_= self.session.run(\
                                        [minimize,self.merged,self.g_step,self.update_moving_average],\
                                        feed_dict={s_t  : s,
                                                a_t  : a,
                                                r_t  : r,
                                                mu_t : mu,
                                                _s_t: s_,
                                                _s_mask_t: s_mask,
                                                task_t: task,
                                                self.var_ep:np.array(report_episode),
                                                self.var_reward:np.array(report_reward),
                                                self.lp: 1.0})
                        if POPART:
                                self.session.run(self.popart)
                        
                        if TRY%5==1:
                                self.train_writer.add_summary(summary,global_step=TRY)
                                self.train_writer.flush()
                        if TRY%100==1:
                                print(TRY)
                                self.model.save("./logs/atari.h5")


                def predict(self, stop):
                        s=[]
                        num=[]
                        for i in range(ACTORS//2):
                                _s,_num = policy_state_push.get()
                                s.append(_s)
                                num.append(_num)

                        s = np.array(s,dtype=np.float32)/255.0

                        with self.default_graph.as_default():
                                r_p, _, _ = self.predict_model.predict([s])

                        for i in range(len(num)):
                                idx = num[i]
                                learner_pipe[idx].send((r_p[i], stop))

                def sync_weight(self):
                        with self.default_graph.as_default():
                                self.predict_model.set_weights(self.model.get_weights())

                

        #---------
        class Optimizer(threading.Thread):
                stop_signal = False

                def __init__(self, taskidx=None):
                        threading.Thread.__init__(self)
                        self.taskidx = taskidx

                def run(self):
                        while not self.stop_signal:
                                brain.optimize()

                def stop(self):
                        self.stop_signal = True
        '''
        class Feeder(threading.Thread):
                stop_signal = False
                

                def __init__(self, taskidx=None):
                        threading.Thread.__init__(self)
                        self.taskidx = taskidx

                def run(self):
                        while not self.stop_signal:
                                brain.feed(self.taskidx)

                def stop(self):
                        self.stop_signal = True
        '''

        class Batcher(threading.Thread):
                stop_signal = False

                def __init__(self):
                        threading.Thread.__init__(self)

                def run(self):
                        while not self.stop_signal:
                                brain.sync_weight()
                                for i in range(N_STEP_UNROLL*2):
                                        brain.predict(self.stop_signal)
                                if self.stop_signal:
                                        break


                def stop(self):
                        self.stop_signal = True

        #-- main
        feeders = []#[Feeder(taskidx=i) for i in range(1)]
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
                if elapsed_time>3600*12:
                        for f in feeders:
                                f.stop()

                        for e in envs:
                                e.terminate()

                        break