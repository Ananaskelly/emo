import tensorflow as tf


class ARXTFModel:

    def __init__(self, context=7, order=2):
        """
        Build ARX model with tf

        :param context: num past feats
        :param order:   num past values
        """
        self.x = None
        self.y_past = None
        self.y = None
        self.y_hat = None
        self.loss = None
        self.opt = None

        self.feat_context = context
        self.order = order

        if self.order == 0:
            self.use_y_past = False
        else:
            self.use_y_past = True

        self.optimizer = tf.train.AdamOptimizer(0.001)
        self.l1_reg = tf.contrib.layers.l1_regularizer(
            scale=0.005, scope=None
        )
        # self.optimizer = tf.train.GradientDescentOptimizer(0.001)

    def build_model(self):
        self.x = tf.placeholder(dtype=float, shape=[None, self.feat_context, 88])
        self.y_past = tf.placeholder(dtype=float, shape=[None, self.order])
        self.y = tf.placeholder(dtype=float, shape=[None])

        self.y_hat = self.regress()

        self.loss = self.get_loss()
        self.opt = self.minimize(self.loss)

    def regress(self):
        # initializer = tf.contrib.layers.xavier_initializer()
        # x_weights = tf.Variable(initializer(shape=[self.feat_context, 88]))
        # y_weights = tf.Variable(initializer(shape=[self.order]))
        x_weights = tf.Variable(tf.truncated_normal(shape=[self.feat_context, 88], stddev=0.1))

        if self.use_y_past:
            y_weights = tf.Variable(tf.truncated_normal(shape=[self.order], stddev=0.01))

        xs = tf.math.multiply(self.x, x_weights)

        if self.use_y_past:
            ys = tf.math.multiply(self.y_past, y_weights)
            return tf.reduce_sum(xs, axis=[1, 2]) + tf.reduce_sum(ys, 1)
        else:
            return tf.reduce_sum(xs, axis=[1, 2])

    def get_loss(self):
        weights = tf.trainable_variables()
        reg_penalty = tf.contrib.layers.apply_regularization(self.l1_reg, weights)
        return tf.losses.mean_squared_error(self.y, self.y_hat) + reg_penalty

    def minimize(self, loss):
        return self.optimizer.minimize(loss)
