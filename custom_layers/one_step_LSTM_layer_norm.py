from keras.layers import *
from keras import backend as K
import tensorflow as tf

class OneStepLSTMLayerNorm(Layer):
    """Cell class for the LSTM layer.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).x
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.]
            (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
    """

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='lecun_normal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 epsilon=1e-8,
                 **kwargs):
        super(OneStepLSTMLayerNorm, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.implementation = implementation
        self.state_size = (self.units, self.units)
        self.output_size = self.units

        self.epsilon = epsilon

    def build(self, input_shape):
        input_shape = input_shape[0]
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = (
            self.recurrent_kernel[:, self.units: self.units * 2])
        self.recurrent_kernel_c = (
            self.recurrent_kernel[:, self.units * 2: self.units * 3])
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]
        
        self.gamma1 = self.add_weight(name='gamma1', shape=(self.units * 4,),
                                     initializer=initializers.Ones(), trainable=True)
        self.beta1 = self.add_weight(name='beta1', shape=(self.units * 4,),
                                     initializer=initializers.Zeros(), trainable=True)

        self.gamma1_i = self.gamma1[:self.units]
        self.gamma1_f = self.gamma1[self.units: self.units * 2]
        self.gamma1_c = self.gamma1[self.units * 2: self.units * 3]
        self.gamma1_o = self.gamma1[self.units * 3:]

        self.beta1_i = self.beta1[:self.units]
        self.beta1_f = self.beta1[self.units: self.units * 2]
        self.beta1_c = self.beta1[self.units * 2: self.units * 3]
        self.beta1_o = self.beta1[self.units * 3:]

        self.gamma2 = self.add_weight(name='gamma2', shape=(self.units * 4,),
                                     initializer=initializers.Ones(), trainable=True)
        self.beta2 = self.add_weight(name='beta2', shape=(self.units * 4,),
                                     initializer=initializers.Zeros(), trainable=True)

        self.gamma2_i = self.gamma2[:self.units]
        self.gamma2_f = self.gamma2[self.units: self.units * 2]
        self.gamma2_c = self.gamma2[self.units * 2: self.units * 3]
        self.gamma2_o = self.gamma2[self.units * 3:]

        self.beta2_i = self.beta2[:self.units]
        self.beta2_f = self.beta2[self.units: self.units * 2]
        self.beta2_c = self.beta2[self.units * 2: self.units * 3]
        self.beta2_o = self.beta2[self.units * 3:]

        self.gamma3 = self.add_weight(name='gamma3', shape=(self.units,),
                                     initializer=initializers.Ones(), trainable=True)
        self.beta3 = self.add_weight(name='beta3', shape=(self.units,),
                                     initializer=initializers.Zeros(), trainable=True)
        self.built = True

    def call(self, input_list):
        inputs = input_list[0]
        h_tm1 = input_list[1]
        c_tm1 = input_list[2]  # previous memory state,previous carry state

        W_x_i = K.dot(inputs, self.kernel_i)
        W_x_f = K.dot(inputs, self.kernel_f)
        W_x_c = K.dot(inputs, self.kernel_c)
        W_x_o = K.dot(inputs, self.kernel_o)

        LN_Wx_i = layer_normal(W_x_i,self.beta1_i,self.gamma1_i, self.epsilon)
        LN_Wx_f = layer_normal(W_x_f,self.beta1_f,self.gamma1_f, self.epsilon)
        LN_Wx_c = layer_normal(W_x_c,self.beta1_c,self.gamma1_c, self.epsilon)
        LN_Wx_o = layer_normal(W_x_o,self.beta1_o,self.gamma1_o, self.epsilon)

        if self.use_bias:
            LN_Wx_i = K.bias_add(LN_Wx_i,self.bias_i)
            LN_Wx_f = K.bias_add(LN_Wx_f,self.bias_f)
            LN_Wx_c = K.bias_add(LN_Wx_c,self.bias_c)
            LN_Wx_o = K.bias_add(LN_Wx_o,self.bias_o)
        
        W_h_i = K.dot(h_tm1, self.recurrent_kernel_i)
        W_h_f = K.dot(h_tm1, self.recurrent_kernel_f)
        W_h_c = K.dot(h_tm1, self.recurrent_kernel_c)
        W_h_o = K.dot(h_tm1, self.recurrent_kernel_o)

        LN_Wh_i = layer_normal(W_h_i,self.beta2_i,self.gamma2_i, self.epsilon)
        LN_Wh_f = layer_normal(W_h_f,self.beta2_f,self.gamma2_f, self.epsilon)
        LN_Wh_c = layer_normal(W_h_c,self.beta2_c,self.gamma2_c, self.epsilon)
        LN_Wh_o = layer_normal(W_h_o,self.beta2_o,self.gamma2_o, self.epsilon)
        
        _i = LN_Wx_i + LN_Wh_i
        _f = LN_Wx_f + LN_Wh_f
        _c = LN_Wx_c + LN_Wh_c
        _o = LN_Wx_o + LN_Wh_o

        i = self.recurrent_activation(_i)
        f = self.recurrent_activation(_f)
        c = f * c_tm1 + i * self.activation(_c)
        o = self.recurrent_activation(_o)

        LN_c = layer_normal(c,self.beta3,self.gamma3, self.epsilon)
        h = o*self.activation(LN_c)
        
        return [h,h,c]

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        output_shape = (input_shape[0], self.units)
        return [output_shape,output_shape,output_shape]

def layer_normal(inputs, beta, gamma, epsilon,method='l2'):
    mean = K.mean(inputs, axis=-1, keepdims=True)
    multiply = 1.0
    if method is 'l1':
        norm = K.mean(K.abs(inputs-mean), axis=-1, keepdims=True)
        multiply = np.sqrt(2.0/np.pi)
        normalized = (inputs-mean)/(norm+epsilon)
        
    elif method is 'l2':
        norm = tf.sqrt(tf.reduce_mean(tf.square(inputs-mean), axis=-1, keepdims=True)+epsilon)
        normalized = (inputs-mean)/(norm+epsilon)

    elif method is 'inf':
        norm = K.max(K.abs(inputs-mean), axis=-1, keepdims=True)
        normalized = (inputs-mean)/(norm+epsilon)

    return  multiply*gamma*normalized + beta
