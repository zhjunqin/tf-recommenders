import tensorflow.compat.v1 as tf

from tensorflow.keras.layers import Layer,InputSpec
from tensorflow.keras import initializers, regularizers, constraints, activations
from tensorflow.python.summary import summary
from tensorflow.python.ops import nn


class FMLayer(Layer):
  """Factorization Machine models feature interactions.
   Input shape:
     3D tensor with shape: ``(batch_size, field_size, embedding_size)``.
   Output shape:
     2D tensor with shape: ``(batch_size, field_size + embedding_size)``.
   References:
     - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
     - part code refer to https://github.com/shenweichen/DeepCTR/
    """
  def __init__(self, **kwargs):

    super(FMLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    if len(input_shape) != 3:
       raise ValueError("Unexpected inputs dimensions % d,\
                        expect to be 3 dimensions" % (len(input_shape)))

    super(FMLayer, self).build(input_shape)  # Be sure to call this somewhere!
    self.built = True

  def call(self, inputs, **kwargs):

    if not self.built:
      self.build(inputs.shape)

    # FM first order, batch_size * field_size
    first_order = tf.reduce_sum(inputs, axis=2)

    # FM second order, batch_size * embedding_size
    square_of_sum = tf.square(tf.reduce_sum(inputs, axis=1))
    sum_of_square = tf.reduce_sum(tf.square(inputs), axis=1)
    second_order = 0.5 * (square_of_sum - sum_of_square)

    # Output, batch_size * (field_size+embedding_size)
    output = tf.concat([first_order, second_order], axis=1)

    return output

  def compute_output_shape(self, input_shape):
    assert input_shape is not None and len(input_shape) == 3

    output_shape = [input_shape[0], input_shape[1] + input_shape[2]]
    output_shape = tuple(output_shape)
    shape = tf.TensorShape([output_shape])
    return shape

