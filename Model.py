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
def build_model():
        l_input = Input(batch_shape=(None,*NUM_STATE))

        l_noise1 = Input(batch_shape=(None, NUM_LSTM, NUM_ACTIONS),name='noise1')
        l_noise2 = Input(batch_shape=(None, NUM_ACTIONS),name='noise2')
        l_noise3 = Input(batch_shape=(None, NUM_LSTM, 1),name='noise3')
        l_noise4 = Input(batch_shape=(None, 1),name='noise4')

        net = Pixel_Shift()(l_input)

        net = Conv2D(16,(8,8),strides=(4, 4))(net)
        net = Activation('relu')(net)

        net = Conv2D(32,(4,4),strides=(2, 2))(net)
        net = Activation('relu')(net)

        net = Flatten()(net)

        net = Dense(256)(net)
        net = Activation('relu')(net)

        out_actions = NoisyDense(NUM_ACTIONS,name='policy_head')([net,l_noise1,l_noise2])
        out_actions = Activation('softmax')(out_actions)

        out_value = PopArt(1,name="popart_value_head")([net,l_noise3,l_noise4])

        model = Model(inputs=[l_input,l_noise1,l_noise2,l_noise3,l_noise4], outputs=[out_actions, out_value])
        
        return model