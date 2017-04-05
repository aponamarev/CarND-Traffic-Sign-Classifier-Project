# Author: Alexander Ponamarev (alex.ponamaryov@gmail.com) 04/30/2017
import tensorflow as tf
from .ClassificationTemplate import ClassificationTemplate

class SmallFilters(ClassificationTemplate):
    def __init__(self, X_placeholders, Y_placeholders, n_classes, probability_density=None):

        ClassificationTemplate.__init__(self, X_placeholders,
                                        Y_placeholders,
                                        n_classes,
                                        probability_density=probability_density)


    def _define_net(self):

        with tf.device('/gpu:0'):

            conv1 = self._conv2d(self.X, [3, 3, 3, 8], bias=True, padding="SAME", name="conv1") #32
            conv2 = self._conv2d(conv1, [3, 3, 8, 16], strides=[1,2,2,1], bias=True, padding="VALID", name="conv2")
            bottleneck3 = self._conv2d(conv2, [1, 1, 16, 4], bias=True, padding="SAME", name="bottleneck3")
            self.activations.append(bottleneck3)
            bn3 = self._batch_norm(bottleneck3, name="bn3") #16

            conv4 = self._conv2d(bn3, [3, 3, 4, 8], bias=False, padding="SAME", name="conv4")
            conv5 = self._conv2d(conv4, [3, 3, 8, 16], strides=[1, 2, 2, 1], bias=True, padding="VALID", name="conv5")
            bottleneck6 = self._conv2d(conv5, [1, 1, 16, 4], bias=True, padding="SAME", name="bottleneck6")
            self.activations.append(bottleneck6)
            bn6 = self._batch_norm(bottleneck6, name="bn6") #8

            conv7 = self._conv2d(bn6, [3, 3, 4, 8], bias=False,
                                 padding="SAME", name="conv7")
            conv8 = self._conv2d(conv7, [3, 3, 8, 16], strides=[1, 2, 2, 1],
                                 bias=True, padding="VALID", name="conv8")
            bottleneck9 = self._conv2d(conv8, [1, 1, 16, 4],
                                       bias=True, padding="SAME", name="bottleneck9")
            self.activations.append(bottleneck9)
            bn9 = self._batch_norm(bottleneck9, name="bn9")  # 4

            dropout = self._drop_out_conv(bn9, "dropout_layer9")

            conv10 = self._conv2d(dropout, [3, 3, 4, 32], strides=[1, 2, 2, 1],
                                  bias=False, padding="SAME", name="conv10") #4
            self.activations.append(conv10)

            conv11 = self._conv2d(conv10, [3, 3, 32, 64], bias=True, padding="VALID", name="conv11") #2
            bn11 = self._batch_norm(conv11, name="bn11")

            with tf.name_scope("feature_map"):
                self.feature_map = tf.squeeze(
                    self._conv2d(bn11,
                                 [2, 2, 64, self._N_CLASSES],
                                 bias=True,
                                 padding="VALID",name="conv"))
