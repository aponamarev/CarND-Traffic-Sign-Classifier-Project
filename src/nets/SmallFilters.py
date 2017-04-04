# Author: Alexander Ponamarev (alex.ponamaryov@gmail.com) 04/30/2017
import tensorflow as tf
from .ClassificationTemplate import ClassificationTemplate

class SmallFilters(ClassificationTemplate):
    def __init__(self, X_placeholders, Y_placeholders, n_classes):

        ClassificationTemplate.__init__(self, X_placeholders, Y_placeholders, n_classes)


    def _define_net(self):

        conv1 = self._conv2d(self.X, [3,3,3,8], bias=True, padding="SAME",name="conv1") #32
        conv2 = self._conv2d(conv1, [3, 3, 8, 16], strides=[1,2,2,1], bias=True, padding="VALID", name="conv2")
        bottleneck3 = self._conv2d(conv2, [1, 1, 16, 4], bias=True, padding="SAME", name="bottleneck3")
        self.activations.append(bottleneck3)
        bn3 = self._batch_norm(bottleneck3, name="bn3") #16

        conv4 = self._conv2d(bn3, [3, 3, 4, 8], bias=False, padding="SAME", name="conv4")
        conv5 = self._conv2d(conv4, [3, 3, 8, 16], strides=[1, 2, 2, 1], bias=True, padding="VALID", name="conv5")
        bottleneck6 = self._conv2d(conv5, [1, 1, 16, 4], bias=True, padding="SAME", name="bottleneck6")
        self.activations.append(bottleneck6)
        bn6 = self._batch_norm(bottleneck6, name="bn6") #8

        dropout = self._drop_out_conv(bn6, "dropout_layer7")

        conv7 = self._conv2d(dropout, [3, 3, 4, 32], strides=[1, 2, 2, 1], bias=False, padding="SAME", name="conv7") #4
        self.activations.append(conv7)

        conv8 = self._conv2d(conv7, [3, 3, 32, 96], bias=True, padding="VALID", name="conv8") #2
        bn8 = self._batch_norm(conv8, name="bn8")  # 8

        with tf.name_scope("feature_map"):
            self.feature_map = tf.squeeze(
                self._conv2d(bn8,
                             [2, 2, 96, self._N_CLASSES],
                             bias=True,
                             padding="VALID",name="conv"))
