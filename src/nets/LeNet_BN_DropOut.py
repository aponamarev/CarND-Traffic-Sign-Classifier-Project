# Author: Alexander Ponamarev (alex.ponamaryov@gmail.com) 04/30/2017
import tensorflow as tf
from .NetTemplate import NetTemplate

class LeNet_BN_DropOut(NetTemplate):
    def __init__(self, input_dict, dropout_placeholder, training_mode_flag):
        img_data = 'img_data'
        labels = 'labels'
        assert (img_data in input_dict.keys()) & (labels in input_dict.keys()),\
            "Incorrect input_dict was provided ({}). Dictionary is expected to contain: {},{}".\
                format(input_dict.keys(), img_data, labels)

        NetTemplate.__init__(self,
                             dropout_rate=dropout_placeholder,
                             training_mode_flag=training_mode_flag,
                             default_activation='elu',
                             dtype=tf.float32)

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
        print("LetNet with DropOut is ready for training!")



    def define_net(self):

        conv1 = self._conv2d(self.X, [5,5,3,6], bias=False, padding="VALID",name="conv1")
        pool1 = self._max_pool(conv1, name="pool1")
        pool1_bn = self._batch_norm(pool1, name="pool1_bn")

        conv2 = self._conv2d(pool1_bn, [5,5, 6, 16], bias=False, padding="VALID", name="conv2")
        pool2 = self._max_pool(conv2, name="pool2")
        pool2_bn = self._batch_norm(pool2, name="pool2_bn")

        fc3 = self._fullyconnected(pool2_bn, 120, name="fc3")
        fc4 = self._fullyconnected(fc3, 84, name="fc4")
        dropout = self._drop_out_fullyconnected(fc4, name="fc4_dropout")

        self.feature_map = self._fullyconnected(dropout, self._N_CLASSES, name="feature_map")

    def define_loss(self):

        with tf.name_scope("cross_entropy"):

            self.total_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.feature_map)
            )

