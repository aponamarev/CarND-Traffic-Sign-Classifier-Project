	# Author: Alexander Ponamarev (alex.ponamaryov@gmail.com) 04/30/2017
import tensorflow as tf
import numpy as np

class NetTemplate(object):
    def __init__(self, training_mode_flag, dropout_keep_rate, default_activation='elu',
                 dtype=tf.float32):
        self.weights = []
        self.size = []
        self.dropout_keep_rate = dropout_keep_rate
        self.is_training_mode = training_mode_flag
        self._default_activation = default_activation
        self._default_activation_summary = 'hist'
        self._dtype = dtype
        self.feature_map=None
        self.total_loss=None
        self.optimization_op=None
        self.activations=[]


    def fit(self, feed_dict):
        raise NotImplementedError("Fit method needs to be defined in the child class")

    def eval(self, feed_dict):
        raise NotImplementedError("Eval method needs to be defined in the child class")

    def infer(self, X_batch):
        raise NotImplementedError("Infer method needs to be defined in the child class")


    def _define_prediction(self):
        self.predict_class_op = None
        raise  NotImplementedError("Prediction method needs to be defined in the child class")

    def _define_accuracy(self):
        self.accuracy_op = None
        raise  NotImplementedError("Accuracy method needs to be defined in the child class")


    def get_size(self):
        """
        Estimates the size of weights of the model.
        
        :return: dictionary with 'parameters' and 'Mb'
        """
        params = np.sum(np.sum(self.size))
        return {'parameters':params, 'Mb': params*self._dtype.size/(2.0**20)}

    def _define_net(self):

        raise NotImplementedError("Feature map encoder was not implemented!")

    def _define_loss(self):
        raise NotImplementedError("Loss estimate was not defined!")

    def _define_optimization_method(self):
        optimizer = tf.train.AdamOptimizer()
        self.optimization_op = optimizer.minimize(self.total_loss)


    def _conv2d(self, inputs, shapes, strides=[1,1,1,1], padding="SAME",name="conv2d", bias=True, dtype=None):
        dtype = dtype or self._dtype
        # Create weights and biases
        with tf.variable_scope(name):
            W = tf.get_variable("W",
                                shapes,
                                dtype=dtype,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())

            self._collect_parameter_stats(W)

            conv = tf.nn.conv2d(inputs, W, strides, padding)

            if bias:
                b_size = shapes[3]
                b_init = tf.zeros(b_size, dtype=dtype)
                b = tf.Variable(b_init, name="b")
                conv = tf.nn.bias_add(conv, b, data_format='NHWC')
                self._collect_parameter_stats(b)

        return self._activation(conv)

    def _fullyconnected(self, inputs, output_channels, name="fully_connected", bias=True, dtype=None):
        dtype = dtype or self._dtype

        X = tf.contrib.layers.flatten(inputs)
        shapes = X.get_shape().as_list()
        shapes = [shapes[1], output_channels]

        # Create weights and biases
        with tf.variable_scope(name):
            W = tf.get_variable("W",
                                shapes,
                                dtype=dtype,
                                initializer=tf.contrib.layers.xavier_initializer())

            self._collect_parameter_stats(W)

            line = tf.matmul(X, W)

            if bias:
                b_size = shapes[1]
                b_init = tf.zeros(b_size, dtype=dtype)
                b = tf.Variable(b_init, name="b")

                self._collect_parameter_stats(b)

                line = tf.nn.bias_add(line, b, data_format='NHWC')

            return self._activation(line)

    def _max_pool(self, inputs, kernel=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name = "max_pool"):
        return tf.nn.max_pool(inputs, kernel, strides, padding=padding, name=name)

    def _avg_pool(self, inputs, kernel=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name = "max_pool"):
        return tf.nn.avg_pool(inputs, kernel, strides, padding=padding, name=name)

    def _batch_norm(self, input, name, trainable=False):
        return tf.contrib.layers.batch_norm(input, trainable=trainable, is_training=self.is_training_mode)

    def _drop_out_fullyconnected(self, input, name):
        return tf.nn.dropout(input, self.dropout_keep_rate, name=name)

    def _drop_out_conv(self, input, name):
        shape = input.get_shape().as_list()
        shape[0] = -1
        with tf.name_scope(name):
            flat = tf.contrib.layers.flatten(input)
            dropout = self._drop_out_fullyconnected(flat, name="dropout")
            conv = tf.reshape(dropout, shape=shape)
        return conv

    def _relu_activation(self, input):

        activation = tf.nn.relu(input)
        self._activation_summary(activation)

        return activation

    def _elu_activation(self, input):

        activation = tf.nn.elu(input)
        self._activation_summary(activation)

        return activation

    def _activation_summary(self, activation):
        type = self._default_activation_summary

        implemented_types = {
            'img': tf.summary.image,
            'hist': tf.summary.histogram
        }

        assert type in implemented_types.keys(), "Incorrect type provided ({}). Only {} types are implemented at the moment". \
            format(type, implemented_types.keys())

        return implemented_types[type]('{}_img'.format(activation.op.name), activation)

    def _activation(self, input, type=None):
        type = type or self._default_activation
        implemented_types = {
            'elu': self._elu_activation,
            'relu': self._relu_activation
        }
        assert type in implemented_types.keys(), "Incorrect type provided ({}). Only {} types are implemented at the moment".\
            format(type, implemented_types.keys())

        return implemented_types[type](input)

    def _collect_parameter_stats(self, variable):
        shape = variable.get_shape().as_list()
        size = np.product(shape, dtype=np.int)
        tf.summary.histogram('{}_hist'.format(variable.op.name), variable)
        self.weights.append(variable)
        self.size.append(size)
