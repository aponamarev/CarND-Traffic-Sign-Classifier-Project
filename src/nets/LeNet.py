# Author: Alexander Ponamarev (alex.ponamaryov@gmail.com) 04/30/2017
import tensorflow as tf
from .NetTemplate import NetTemplate

class LeNet(NetTemplate):
    def __init__(self, input_dict):
        img_data = 'img_data'
        labels = 'labels'
        assert (img_data in input_dict.keys()) & (labels in input_dict.keys()),\
            "Incorrect input_dict was provided ({}). Dictionary is expected to contain: {},{}".\
                format(input_dict.keys(), img_data, labels)

        NetTemplate.__init__(self,
                             dropout_rate=tf.constant(1.0, dtype=tf.float32),
                             training_mode_flag=False,
                             default_activation='elu', dtype=tf.float32)

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
        print("LetNet is ready for training!")



    def define_net(self):

        conv1 = self._conv2d(self.X, [5,5,3,6], bias=True, padding="VALID",name="conv1")
        pool1 = self._max_pool(conv1, name="pool1")

        conv2 = self._conv2d(pool1, [5,5, 6, 16], bias=True, padding="VALID", name="conv2")
        pool2 = self._max_pool(conv2, name="pool2")

        fc3 = self._fullyconnected(pool2, 120, name="fc3")
        fc4 = self._fullyconnected(fc3, 84, name="fc4")

        self.feature_map = self._fullyconnected(fc4, self._N_CLASSES, name="feature_map")

    def define_loss(self):

        with tf.name_scope("cross_entropy"):

            self.total_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.feature_map)
            )

