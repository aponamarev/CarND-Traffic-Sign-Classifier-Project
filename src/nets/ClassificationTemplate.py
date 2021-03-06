# Author: Alexander Ponamarev (alex.ponamaryov@gmail.com) 04/30/2017
import tensorflow as tf
from .NetTemplate import NetTemplate

class ClassificationTemplate(NetTemplate):
    def __init__(self, X_placeholders, Y_placeholders, n_classes, default_activation='elu',
                 dtype=tf.float32, probability_density = None, trainset_mean=None, gpu="/gpu:0"):

        self._gpu = gpu
        self._img_mean = trainset_mean

        tf.add_to_collection('inputs', X_placeholders)
        tf.add_to_collection('inputs', Y_placeholders)

        with tf.name_scope('inputs'):
            self.X = X_placeholders
            tf.summary.image("imgs", self.X, max_outputs=6)
            self.labels = Y_placeholders

            X_float = tf.divide(self.X, 255.0)
            if self._img_mean is not None:
                self.X_norm = tf.subtract(X_float, self._img_mean, name="X_norm")
            else:
                shape = X_float.get_shape().as_list()
                shape[0] = -1
                X_flat = tf.contrib.layers.flatten(X_float)
                self.X_norm = tf.reshape(tf.subtract(X_flat, tf.reduce_mean(X_flat, axis=1, keep_dims=True, name="X_norm")),
                                         shape)
            tf.summary.image("norm_imgs", self.X_norm, max_outputs=6)


        self._N_CLASSES = n_classes

        if probability_density is None:
            self.pdf=None
        else:
            self.pdf = tf.constant(probability_density, shape=[self._N_CLASSES],
                                   dtype=dtype, name='probability_density_function')

        NetTemplate.__init__(self,
                             dropout_keep_rate=tf.placeholder(dtype=tf.float32, shape=[], name="dropout_keep_prob"),
                             training_mode_flag=tf.placeholder(dtype=tf.bool, shape=[], name="is_training_phase"),
                             default_activation=default_activation,
                             dtype=dtype)

        self._assemble()

    def _assemble(self):
        self._define_net()
        print("Autoencoder was successfully created.")
        self._define_loss()
        print("Loss definition was successfully created.")
        self._define_optimization_method()
        print("Optimization function was initialized.")
        self._define_prediction()
        self._define_accuracy()
        print("{} is ready for training!".format(type(self).__name__))


    def _define_loss(self):

        # Note: When is_training is True the moving_mean and moving_variance need to be updated, by default the update_ops are placed in tf.GraphKeys.UPDATE_OPS so they need to be added as a dependency to the train_op, example:
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # if update_ops:
        #     updates = tf.group(*update_ops)
        #     total_loss = control_flow_ops.with_dependencies([updates], total_loss)
        # Reference: http://ruishu.io/2016/12/27/batchnorm/

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.device(self._gpu):
            with tf.control_dependencies(update_ops):
                # Ensures that we execute the update_ops before performing the train_step
                with tf.name_scope("cross_entropy_loss"):
                    if self.pdf is None:
                        self.total_loss = tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                           logits=self.feature_map)
                        )
                    else:
                        P_of_x = tf.nn.softmax(logits=self.feature_map)
                        P_of_x_given_PDF = tf.divide(P_of_x, self.pdf)

                        self.total_loss = tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                           logits=P_of_x_given_PDF)
                        )

        tf.summary.scalar("total_loss", self.total_loss)

    def _define_prediction(self):
        assert self.feature_map is not None, "Error: Feature map wasn't defined."
        with tf.device(self._gpu):
            with tf.name_scope("class_prediction"):
                self.probability_op = tf.nn.softmax(self.feature_map, name="probability")
                self.predict_class_op = tf.arg_max(self.probability_op, 1, name="label")
                tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self.predict_class_op)

    def _define_accuracy(self):
        with tf.device(self._gpu):
            self.accuracy_op = tf.reduce_mean(
                tf.cast(tf.equal(self.predict_class_op, self.labels, name="predict_accuracy"),
                        dtype=tf.float32)
            )
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self.accuracy_op)

        tf.summary.scalar("accuracy", self.accuracy_op)

    def eval(self, X_batch, Y_batch):
        sess = tf.get_default_session()
        accuracy = sess.run(self.accuracy_op, feed_dict={self.X: X_batch,
                                                         self.labels: Y_batch,
                                                         self.is_training_mode: False,
                                                         self.dropout_keep_rate: 1.0})
        return accuracy

    def fit(self, X_batch, Y_batch, dropout_keep_prob=0.75):
        sess = tf.get_default_session()
        _, accuracy = sess.run([self.optimization_op, self.accuracy_op],
                               feed_dict={self.X: X_batch,
                                          self.labels: Y_batch,
                                          self.is_training_mode: True,
                                          self.dropout_keep_rate: dropout_keep_prob}
                               )
        return accuracy

    def infer(self, X_batch):
        sess = tf.get_default_session()
        probability, predict_class = sess.run([self.probability_op, self.predict_class_op],
                                              feed_dict={self.X: X_batch,
                                                         self.is_training_mode: False,
                                                         self.dropout_keep_rate: 1.0})
        return predict_class, probability



