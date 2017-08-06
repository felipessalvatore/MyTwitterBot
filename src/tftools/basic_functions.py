import tensorflow as tf


def init_wb(shape, name):
    """
    Function initialize one matrix of weights and one bias vector.

    :type shape: tuple
    :type name: str
    :rtype: dictionary
    """
    Winit = tf.truncated_normal(shape, mean=0, stddev=0.1)
    binit = tf.zeros(shape[-1])
    layer = {}
    layer["weights"] = tf.get_variable(name + "/weights",
                                       dtype=tf.float32,
                                       initializer=Winit)
    layer["bias"] = tf.get_variable(name + "/bias",
                                    dtype=tf.float32,
                                    initializer=binit)
    return layer


def affine_transformation(input_tensor, layer):
    """
    Function that applies a affine transformation
    in the input tensor using the variables
    from the dict layer.

    :type input_tensor: tf tensor
    :type layer: dictionary
    :rtype: tf tensor
    """
    return tf.add(tf.matmul(input_tensor, layer['weights']),
                  layer['bias'])
