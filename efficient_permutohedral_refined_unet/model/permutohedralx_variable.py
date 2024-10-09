import numpy as np
import tensorflow as tf


class PermutohedralXVariable(tf.Module):
    def __init__(self, N: tf.int32, d: tf.int32, name=None):
        super().__init__(name)

        self.N = tf.constant(N, dtype=tf.int32)
        self.d = tf.constant(d, dtype=tf.int32)

        self.os = tf.Variable(
            tf.constant(
                0,
                dtype=tf.int32,
                shape=[
                    self.N * (self.d + 1),
                ],
            ),
            trainable=False,
        )
        self.ws = tf.Variable(
            tf.constant(
                0,
                dtype=tf.float32,
                shape=[
                    self.N * (self.d + 1),
                ],
            ),
            trainable=False,
        )
        self.M = tf.Variable(
            tf.constant(0, dtype=tf.int32),
            trainable=False,
        )
