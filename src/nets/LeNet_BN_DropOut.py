# Author: Alexander Ponamarev (alex.ponamaryov@gmail.com) 04/30/2017
from .ClassificationTemplate import ClassificationTemplate

class LeNet_BN_DropOut(ClassificationTemplate):

    def __init__(self, X_placeholders, Y_placeholders, n_classes):

        ClassificationTemplate.__init__(self, X_placeholders, Y_placeholders, n_classes)


    def _define_net(self):

        conv1 = self._conv2d(self.X, [5,5,3,6], bias=False, padding="VALID",name="conv1")
        pool1 = self._max_pool(conv1, name="pool1")
        pool1_bn = self._batch_norm(pool1, name="pool1_bn")
        self.activations.append(pool1_bn)

        conv2 = self._conv2d(pool1_bn, [5,5, 6, 16], bias=False, padding="VALID", name="conv2")
        pool2 = self._max_pool(conv2, name="pool2")
        pool2_bn = self._batch_norm(pool2, name="pool2_bn")
        self.activations.append(pool2_bn)

        fc3 = self._fullyconnected(pool2_bn, 120, name="fc3")
        fc4 = self._fullyconnected(fc3, 84, name="fc4")
        dropout = self._drop_out_fullyconnected(fc4, name="fc4_dropout")

        self.feature_map = self._fullyconnected(dropout, self._N_CLASSES, name="feature_map")

