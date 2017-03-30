# Author: Alexander Ponamarev (alex.ponamaryov@gmail.com) 04/30/2017
import tensorflow as tf
import numpy as np

class NetTemplate(object):
    def __init__(self, input_dict):
        self.inputs = input_dict
        self.weights = []
        self.size = []

    def get_size(self):
        return np.sum(np.sum(self.size))

    def define_net(self):

        img_batch = self.inputs['X']

        conv = self._conv2d(img_batch, [3,3,3,16], bias=True, name="Layer1")
        conv2 = self._conv2d(conv, [3,3,16, 64], bias=False, name="Layer2")

        return conv

    def _conv2d(self, inputs, shapes, strides=[1,1,1,1], padding="SAME",name="conv2d", bias=True, dtype=tf.float32):
        # Track stats
        parameters = []
        size = []
        # Create weights and biases
        with tf.variable_scope(name):
            W = tf.get_variable("W",
                                shapes,
                                dtype=dtype,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())

            parameters.append(W)
            size.append(np.prod(shapes, dtype = np.int))

            conv = tf.nn.conv2d(inputs, W, strides, padding)

            if bias:
                b_size = shapes[3]
                b_init = tf.zeros(b_size, dtype=dtype)
                b = tf.Variable(b_init, name="b")
                conv = tf.nn.bias_add(conv, b, data_format='NHWC')
                parameters.append(b)
                size.append(b_size)

        self.weights.append(parameters)
        self.size.append(size)

        return conv

if __name__ == '__main__':

    X = tf.placeholder(dtype=tf.float32, shape=[None, 32,32, 3], name="X")

    input_dict = {"X": X}

    test_net = NetTemplate(input_dict)

    conv_layer = test_net.define_net()
    print("Model size:", test_net.get_size())

    print("test was successful!")