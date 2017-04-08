# Author: Alexander Ponamarev (alex.ponamaryov@gmail.com) 04/30/2017
import tensorflow as tf
from .ClassificationTemplate import ClassificationTemplate

class LeNet(ClassificationTemplate):
    def __init__(self, X_placeholders, Y_placeholders, n_classes, probability_density=None):

        ClassificationTemplate.__init__(self, X_placeholders,
                                        Y_placeholders,
                                        n_classes,
                                        probability_density=probability_density)


    def _define_net(self):
        with tf.device("/gpu:0"):

            conv1 = self._conv2d(self.X, [5, 5, 3, 6], bias=True, padding="VALID", name="conv1")
            pool1 = self._max_pool(conv1, name="pool1")
            self.activations.append(pool1)

            conv2 = self._conv2d(pool1, [5,5, 6, 16], bias=True, padding="VALID", name="conv2")
            pool2 = self._max_pool(conv2, name="pool2")
            self.activations.append(pool2)

            fc3 = self._fullyconnected(pool2, 120, name="fc3")
            fc4 = self._fullyconnected(fc3, 84, name="fc4")

            self.feature_map = self._fullyconnected(fc4, self._N_CLASSES, name="feature_map")
