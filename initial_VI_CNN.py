import numpy as np
import tensorflow as tf
from utils import conv2d_flipkernel

def VI_Block(X, S1, S2, config):
    n    = config.n    # Number of value iterations performed
    ch_i = config.ch_i # Channels in input layer
    state_batch_size = config.statebatchsize # N+1 state inputs for each channel

    bias  = tf.Variable(np.random.randn(1, 1, 1, ch_i)    * 0.01, dtype=tf.float32)
    # weights from inputs to q layer (~reward in Bellman equation)
    w    = tf.Variable(np.random.randn(1, 1, ch_i, 1)    * 0.01, dtype=tf.float32)
    # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
    w_fb  = tf.Variable(np.random.randn(1, 1, ch_i, 1)    * 0.01, dtype=tf.float32)
    w_o   = tf.Variable(np.random.randn(ch_i, 8)          * 0.01, dtype=tf.float32)

    # initial conv layer over reward prior
    h = conv2d_flipkernel(X, w, name="h0") + bias

    r = conv2d_flipkernel(h, name="r")
    q = conv2d_flipkernel(r, w, name="q")
    v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    for i in range(0, n-1):
        rv = tf.concat([r, v], 3)
        wwfb = tf.concat([w, w_fb], 2)
        q = conv2d_flipkernel(rv, wwfb, name="q")
        v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    # do one last convolution
    q = conv2d_flipkernel(tf.concat([r, v], 3),
                          tf.concat([w, w_fb], 2), name="q")


    q = tf.transpose(q, perm=[0, 3, 1, 2])

    # Select the conv-net channels at the state position (S1,S2).
    # This intuitively corresponds to each channel representing an action, and the convnet the Q function.

    # TODO: performance can be improved here by substituting expensive
    #       transpose calls with better indexing for gather_nd
    bs = tf.shape(q)[0]
    rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
    q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")

    # add logits
    logits = tf.matmul(q_out, w_o)
    # softmax output weights
    output = tf.nn.softmax(logits, name="output")
    return logits, output

# similar to the normal VI_Block except there are separate weights for each q layer
def VI_Untied_Block(X, S1, S2, config):
    n    = config.n    # Number of value iterations performed
    ch_i = config.ch_i # Channels in input layer
    state_batch_size = config.statebatchsize # n+1 state inputs for each channel

    bias   = tf.Variable(np.random.randn(1, 1, ch_i, 1)    * 0.01, dtype=tf.float32)
    # weights from inputs to q layer (~reward in Bellman equation)
    w     = tf.Variable(np.random.randn(3, 3, ch_i, 1) * 0.01, dtype=tf.float32)
    # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
    w_fb_l = [tf.Variable(np.random.randn(3, 3, ch_i, 1)   * 0.01, dtype=tf.float32) for i in range(0,n)]
    w_o    = tf.Variable(np.random.randn(ch_i, 8)          * 0.01, dtype=tf.float32)

    # initial conv layer over image+reward prior
    h = conv2d_flipkernel(X, w, name="h0") + bias

    r = conv2d_flipkernel(h, w, name="r")
    q = conv2d_flipkernel(r, w[0], name="q")
    v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    for i in range(0, n-1):
        rv = tf.concat([r, v], 3)
        wwfb = tf.concat([w[i+1], w_fb_l[i]], 2)
        q = conv2d_flipkernel(rv, wwfb, name="q")
        v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    # do one last convolution
    q = conv2d_flipkernel(tf.concat([r, v], 3),
                          tf.concat([w[n], w_fb_l[n-1]], 2), name="q")

    # Since we are selecting one channels, it becomes easier to work with the tensor
    q = tf.transpose(q, perm=[0, 3, 1, 2])

    # Select the conv-net channels at the state position (S1,S2).
    # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
    # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
    # TODO: performance can be improved here by substituting expensive
    #       transpose calls with better indexing for gather_nd
    bs = tf.shape(q)[0]
    rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
    q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")

    # add logits
    logits = tf.matmul(q_out, w_o)
    # softmax output weights
    output = tf.nn.softmax(logits, name="output")
    return logits, output
