import numpy as np
import tensorflow as tf

def flipkernel(kern):
    return kern[(slice(None, None, -1),) * 2 + (slice(None), slice(None))]

def conv2d_flipkernel(x, n, name=None):
    return tf.nn.conv2d(x, flipkernel(n), name=name,
                        strides=(1, 1, 1, 1), padding='SAME')