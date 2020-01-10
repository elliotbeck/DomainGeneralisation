import tensorflow as tf

def tf_bernoulli(p, size):
    return tf.cast([tf.random.uniform([size]) < p], dtype=tf.float32)


def tf_xor(a, b):
    return tf.abs((a-b)) # Assumes both inputs are either 0 or 1