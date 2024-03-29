from keras import backend as K
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
import numpy as np
import tensorflow as tf
class Layer_Ell(Layer):

    def __init__(self,
                axis=-1,
                epsilon=1e-12,
                method='l2',
                center = True,
                scale = True,
                beta_initializer='zeros',
                gamma_initializer='ones',
                beta_regularizer=None,
                gamma_regularizer=None,
                beta_constraint=None,
                gamma_constraint=None,
                **kwargs):
        super(Layer_Ell, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
        self.method = method
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)


    def build(self, input_shape):
        # Prepare broadcasting shape.
        dim = input_shape[self.axis]
        shape = (dim,)

        #if ndim>2:
        #    self.reduction_axes = self.reduction_axes[:-1]
        if self.scale:
            self.gamma = self.add_weight(name='gamma', shape=shape,
                                     initializer=self.gamma_initializer, trainable=True)
        else:
            self.gamma = 1.0
        if self.center:
            self.beta = self.add_weight(name='beta', shape=shape,
                                     initializer=self.beta_initializer, trainable=True)
        else:
            self.beta = 0.0
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        self.reduction_axes = list(range(len(input_shape)))
        del self.reduction_axes[0]
        mean = K.mean(inputs, axis=self.reduction_axes, keepdims=True)

        if self.method is 'l1':
            norm = tf.reduce_mean(K.abs(inputs-mean), axis=self.reduction_axes, keepdims=True)

        elif self.method is 'l2':
            norm = tf.sqrt(tf.reduce_mean(tf.square(inputs-mean), axis=self.reduction_axes, keepdims=True))

        elif self.method is 'inf':
            norm = K.max(K.abs(inputs-mean), axis=self.reduction_axes, keepdims=True)
        
        normalized = (inputs-mean)/(norm+self.epsilon)

        ans = self.gamma*normalized + self.beta
        return ans

    def compute_output_shape(self, input_shape):
        return input_shape