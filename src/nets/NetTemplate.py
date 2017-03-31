# Author: Alexander Ponamarev (alex.ponamaryov@gmail.com) 04/30/2017
import tensorflow as tf
import numpy as np

class NetTemplate(object):
    def __init__(self, input_dict, default_activation='elu', dtype=tf.float32):
        self.inputs = input_dict
        self.weights = []
        self.size = []
        self._default_activation = default_activation
        self._dtype = dtype

    def get_size(self):
        params = np.sum(np.sum(self.size))
        return {'parameters':params, 'Mb': params*self._dtype.size/(2.0**20)}

    def define_net(self):

        img_batch = self.inputs['X']

        net = self._conv2d(img_batch, [3,3,3,16], bias=True, name="Layer1")
        net = self._batch_norm(net)
        net = self._conv2d(net, [3,3,16, 64], bias=False, name="Layer2")
        net = self._batch_norm(net)
        net = self._max_pool(net)
        net = self._batch_norm(net)
        net = self._avg_pool(net)
        net = self._batch_norm(net)
        net = self._fullyconnected(net,10)

        return net

    def _conv2d(self, inputs, shapes=[1,3,3,1], strides=[1,1,1,1], padding="SAME",name="conv2d", bias=True, dtype=None):
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

    def _batch_norm(self, input):
        return tf.layers.batch_normalization(input)

    def _relu_activation(self, input):

        activation = tf.nn.relu(input)
        tf.summary.histogram('{}_hist'.format(activation.op.name), activation)

        return activation

    def _elu_activation(self, input):

        activation = tf.nn.elu(input)
        tf.summary.histogram('{}_hist'.format(activation.op.name), activation)

        return activation

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


if __name__ == '__main__':

    X = tf.placeholder(dtype=tf.float32, shape=[None, 32,32, 3], name="X")

    input_dict = {"X": X}

    test_net = NetTemplate(input_dict)

    conv_layer = test_net.define_net()
    print("Model size:", test_net.get_size())
    shape = conv_layer.get_shape().as_list()
    print("Model shape:", shape)

    print("test was successful!")