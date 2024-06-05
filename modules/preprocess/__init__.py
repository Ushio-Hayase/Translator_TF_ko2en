import tensorflow as tf
import tensorflow.keras.backend as B
import tensorflow.keras as K

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis,tf.newaxis, :] 

def create_look_ahead_mask(x : tf.Tensor):
    size_1 = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size_1, size_1)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)
