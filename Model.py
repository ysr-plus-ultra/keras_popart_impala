import keras
import json
import keras.backend.tensorflow_backend as K
from keras.layers import Dense,Activation,Lambda,Flatten,Conv2D,BatchNormalization,Dropout,Concatenate
from keras.models import Model,Input
from custom_layers.NoisyDense import NoisyDense
from custom_layers.layer_ell_norm import Layer_Ell
from custom_layers.one_step_LSTM import OneStepLSTM
from custom_layers.one_step_LSTM_layer_norm import OneStepLSTMLayerNorm
from custom_layers.WN_Dense import WN_Dense
from custom_layers.WN_Conv2d import WN_Conv2D
from custom_layers.popart_layer import PopArt
from custom_layers.pixel_shift import Pixel_Shift

with open('setting.json') as json_file:  
        SETTING = json.load(json_file)
NUM_STATE = SETTING['N_STATE']
NUM_ACTIONS = SETTING['N_ACTIONS']
NUM_LSTM = SETTING['N_LSTM']
NUM_TASK = SETTING['N_TASK']
def build_model():
        l_input = Input(batch_shape=(None,*NUM_STATE))

        net = Conv2D(32,(8,8),strides=(4,4),padding='valid',kernel_initializer='glorot_uniform')(l_input)
        net = Activation('relu')(net)

        net = Conv2D(64,(4,4),strides=(2,2),padding='valid',kernel_initializer='glorot_uniform')(net)
        net = Activation('relu')(net)

        net = Flatten()(net)

        net = Dense(512,kernel_initializer='glorot_uniform')(net)
        net = Activation('relu')(net)

        out_actions = Dense(NUM_ACTIONS,name='policy_head')(net)
        out_actions = Activation('softmax')(out_actions)

        out_value, unnomalized_value = PopArt(NUM_TASK,name="popart_value_head")(net)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value, unnomalized_value])
        
        return model