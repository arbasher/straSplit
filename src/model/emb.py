import math

import tensorflow as tf


class Embedding(object):
    def __init__(self, size: int = 20, dimension: int = 50, embeddings=None, use_truncated_normal: bool = False):
        if size == 0 or dimension == 0:
            return
        if embeddings is None:
            embeddings = self.__initialize_weights(size=size, dimension=dimension,
                                                   use_truncated_normal=use_truncated_normal)
        self.embeddings = embeddings

    def __initialize_weights(self, size: int = 20, dimension: int = 50, use_truncated_normal: bool = False):
        if not use_truncated_normal:
            bound = math.sqrt(6. / dimension)
            embeddings = tf.random.uniform([size, dimension], minval=-bound,
                                           maxval=bound, dtype=tf.dtypes.float32)
        else:
            bound = 1.0 / math.sqrt(dimension)
            embeddings = tf.random.truncated_normal([size, dimension], mean=0.0, stddev=bound,
                                                    dtype=tf.dtypes.float32)
        return embeddings
