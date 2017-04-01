# Author: Alexander Ponamarev (alex.ponamaryov@gmail.com) 04/30/2017
import tensorflow as tf
from .NetTemplate import NetTemplate

class SmallFilters(NetTemplate):
    def __init__(self, input_dict, dropout_placeholder):
        img_data = 'img_data'
        labels = 'labels'
        assert (img_data in input_dict.keys()) & (labels in input_dict.keys()),\
            "Incorrect input_dict was provided ({}). Dictionary is expected to contain: {},{}".\
                format(input_dict.keys(), img_data, labels)

        NetTemplate.__init__(self,
                             default_activation='elu',
                             dtype=tf.float32,
                             dropout_rate=dropout_placeholder)

        self.X = input_dict[img_data]
        self.Y = input_dict[labels]
        self._N_CLASSES = self.Y.get_shape().as_list()[1]

        self.assemble()

    def assemble(self):
        self.define_net()
        print("Autoencoder was successfully created.")
        self.define_loss()
        print("Loss definition was successfully created.")
        self.define_optimization_method()
        print("Optimization function was initialized.")
        print("SmallFiltersNet is ready for training!")



    def define_net(self):

        conv1 = self._conv2d(self.X, [3,3,3,8], bias=True, padding="SAME",name="conv1") #32
        conv2 = self._conv2d(conv1, [3, 3, 8, 16], strides=[1,2,2,1], bias=True, padding="VALID", name="conv2")
        bottleneck3 = self._conv2d(conv2, [1, 1, 16, 4], bias=True, padding="SAME", name="bottleneck3")
        bn3 = self._batch_norm(bottleneck3, name="bn3") #16

        conv4 = self._conv2d(bn3, [3, 3, 4, 16], bias=False, padding="SAME", name="conv4")
        conv5 = self._conv2d(conv4, [3, 3, 16, 32], strides=[1, 2, 2, 1], bias=True, padding="VALID", name="conv5")
        bottleneck6 = self._conv2d(conv5, [1, 1, 32, 16], bias=True, padding="SAME", name="bottleneck6")
        bn6 = self._batch_norm(bottleneck6, name="bn6") #8

        dropout = self._drop_out_conv(bn6, "dropout_layer6")

        conv7 = self._conv2d(dropout, [3, 3, 16, 64], strides=[1, 2, 2, 1], bias=False, padding="SAME", name="conv7") #4
        conv8 = self._conv2d(conv7, [3, 3, 64, 128], bias=True, padding="VALID", name="conv8") #2
        with tf.name_scope("feature_map"):
            self.feature_map = tf.squeeze(
                self._conv2d(conv8,
                             [2, 2, 128, self._N_CLASSES],
                             bias=True,
                             padding="VALID",name="conv"))

    def define_loss(self):

        with tf.name_scope("cross_entropy"):

            self.total_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.feature_map)
            )

