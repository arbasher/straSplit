import tensorflow as tf

import config


class Generator(object):
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        with tf.compat.v1.variable_scope('generator'):
            self.embedding_matrix = tf.compat.v1.get_variable(name="embedding",
                                                              shape=self.node_emd_init.shape,
                                                              initializer=tf.compat.v1.constant_initializer(
                                                                  self.node_emd_init),
                                                              trainable=True)
            self.bias_vector = tf.compat.v1.Variable(tf.compat.v1.zeros([self.n_node]))

        self.node_id = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.node_neighbor_id = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.reward = tf.compat.v1.placeholder(tf.float32, shape=[None])

        self.all_score = tf.compat.v1.matmul(self.embedding_matrix, self.embedding_matrix,
                                             transpose_b=True) + self.bias_vector
        self.node_embedding = tf.compat.v1.nn.embedding_lookup(self.embedding_matrix,
                                                               self.node_id)  # batch_size * n_embed
        self.node_neighbor_embedding = tf.compat.v1.nn.embedding_lookup(self.embedding_matrix, self.node_neighbor_id)
        self.bias = tf.compat.v1.gather(self.bias_vector, self.node_neighbor_id)
        self.score = tf.compat.v1.reduce_sum(self.node_embedding * self.node_neighbor_embedding, axis=1) + self.bias
        self.prob = tf.compat.v1.clip_by_value(tf.compat.v1.nn.sigmoid(self.score), 1e-5, 1)

        self.loss = -tf.compat.v1.reduce_mean(tf.compat.v1.log(self.prob) * self.reward) + config.lambda_gen * (
                tf.compat.v1.nn.l2_loss(self.node_neighbor_embedding) + tf.compat.v1.nn.l2_loss(self.node_embedding))
        optimizer = tf.compat.v1.train.AdamOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)
