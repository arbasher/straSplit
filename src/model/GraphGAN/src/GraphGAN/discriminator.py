import tensorflow as tf

import config


class Discriminator(object):
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        with tf.compat.v1.variable_scope('discriminator'):
            self.embedding_matrix = tf.compat.v1.get_variable(name="embedding",
                                                              shape=self.node_emd_init.shape,
                                                              initializer=tf.compat.v1.constant_initializer(
                                                                  self.node_emd_init),
                                                              trainable=True)
            self.bias_vector = tf.compat.v1.Variable(tf.compat.v1.zeros([self.n_node]))

        self.node_id = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.node_neighbor_id = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.label = tf.compat.v1.placeholder(tf.float32, shape=[None])

        self.node_embedding = tf.compat.v1.nn.embedding_lookup(self.embedding_matrix, self.node_id)
        self.node_neighbor_embedding = tf.compat.v1.nn.embedding_lookup(self.embedding_matrix, self.node_neighbor_id)
        self.bias = tf.compat.v1.gather(self.bias_vector, self.node_neighbor_id)
        self.score = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.node_embedding, self.node_neighbor_embedding),
                                             axis=1) + self.bias

        self.loss = tf.compat.v1.reduce_sum(
            tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=self.label,
                                                              logits=self.score)) + config.lambda_dis * (
                            tf.compat.v1.nn.l2_loss(self.node_neighbor_embedding) +
                            tf.compat.v1.nn.l2_loss(self.node_embedding) +
                            tf.compat.v1.nn.l2_loss(self.bias))
        optimizer = tf.compat.v1.train.AdamOptimizer(config.lr_dis)
        self.d_updates = optimizer.minimize(self.loss)
        self.score = tf.compat.v1.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)
        self.reward = tf.compat.v1.log(1 + tf.compat.v1.exp(self.score))
