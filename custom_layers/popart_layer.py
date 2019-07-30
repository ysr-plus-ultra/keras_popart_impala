from keras.layers import *
from keras import backend as K
import tensorflow as tf

class PopArt(Layer):
    

    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='lecun_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 sigma_init = 0.05,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(PopArt, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.sigma_init = sigma_init


    def build(self, input_shape):
        input_shape = input_shape[0]
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.sigmaW = self.add_weight(shape=(input_dim, self.units),
                                      initializer=initializers.Constant(value=self.sigma_init),
                                      name='sigmaW')

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.sigmab = self.add_weight(shape=(self.units,),
                                        initializer=initializers.Constant(value=self.sigma_init),
                                        name='sigmab')
        else:
            self.bias = None

        self.mu = tf.Variable(0.0,trainable=False,name='popart_mu')
        self.nu = tf.Variable(1.0,trainable=False,name='popart_nu')
        self.inv_sigma = tf.rsqrt(self.nu-tf.square(self.mu))
        self._non_trainable_weights.append(self.mu)
        self._non_trainable_weights.append(self.nu)

        self.built = True

    def call(self, inputs):
        batch = inputs[0]
        noise_p = inputs[1]*self.inv_sigma
        noise_q = inputs[2]*self.inv_sigma
        output = K.dot(batch, self.kernel) + tf.einsum('ij,ijk->ik',batch,(self.sigmaW*noise_p))
        if self.use_bias:
            output = K.bias_add(output, self.bias) + (self.sigmab*noise_q)

        return output

    def compute_output_shape(self, input_shape):
        #assert input_shape and len(input_shape) >= 2
        #assert input_shape[-1]
        input_shape = input_shape[0]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(PopArt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
