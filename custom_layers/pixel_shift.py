from keras import backend as K
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
import numpy as np
import tensorflow as tf
class Pixel_Shift(Layer):

    def __init__(self,         
                **kwargs):
        super(Pixel_Shift, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mean = self.add_weight(name='pixel_mean', shape=input_shape[1:-1],
                                     initializer=initializers.Zeros(), trainable=False)
        self.built = True

    def call(self, inputs):
        return inputs-tf.expand_dims(self.mean,axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape