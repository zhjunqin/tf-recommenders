# Copyright 2021 The TensorFlow Recommenders Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements `Cross` Layer, the cross layer in Deep & Cross Network (DCN).
Code from https://github.com/tensorflow/recommenders
"""

from typing import Union, Text, Optional

import tensorflow.compat.v1 as tf

from tensorflow.keras.layers import Layer,InputSpec
from tensorflow.keras import initializers, regularizers, constraints, activations
from tensorflow.python.summary import summary
from tensorflow.python.ops import nn


class CrossLayer(tf.keras.layers.Layer):
  """Cross Layer in Deep & Cross Network to learn explicit feature interactions.
    References:
        1. [R. Wang et al.](https://arxiv.org/pdf/1708.05123.pdf)
    Example:
        ```python
        # after embedding layer in a functional model:
        input = tf.keras.Input(shape=(None,), name='index', dtype=tf.int64)
        x0 = tf.keras.layers.Embedding(input_dim=32, output_dim=6)
        x1 = CrossLayer()(x0, x0)
        x2 = CrossLayer()(x0, x1)
        logits = tf.keras.layers.Dense(units=10)(x2)
        model = tf.keras.Model(input, logits)
        ```
    Args:
        kernel_initializer: Initializer to use on the kernel matrix.
        bias_initializer: Initializer to use on the bias vector.
        kernel_regularizer: Regularizer to use on the kernel matrix.
        bias_regularizer: Regularizer to use on bias vector.
    Input shape: A tuple of 2 (batch_size, `input_dim`) dimensional inputs.
    Output shape: A single (batch_size, `input_dim`) dimensional output.
  """

  def __init__(
      self,
      kernel_initializer: Union[
          Text, tf.keras.initializers.Initializer] = "truncated_normal",
      bias_initializer: Union[Text,
                              tf.keras.initializers.Initializer] = "zeros",
      kernel_regularizer: Union[Text, None,
                                tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Union[Text, None,
                              tf.keras.regularizers.Regularizer] = None,
      **kwargs):

    super(CrossLayer, self).__init__(**kwargs)

    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._input_dim = None

    self._supports_masking = True

  def build(self, input_shape):
    last_dim = int(input_shape[-1])

    self._weight = self.add_weight(
        name='weight',
        shape=(last_dim, 1),
        initializer=self._kernel_initializer,
        regularizer=self._kernel_regularizer,
        trainable=True)

    self._bias = self.add_weight(
        name='bias',
        shape=(1, last_dim),
        initializer=self._bias_initializer,
        regularizer=self._bias_regularizer,
        trainable=True)

    self.built = True

  def call(self, x0: tf.Tensor, x: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Computes the feature cross.
    Args:
      x0: The input tensor
      x: Optional second input tensor. If provided, the layer will compute
        crosses between x0 and x; if not provided, the layer will compute
        crosses between x0 and itself.
    Returns:
     Tensor of crosses.
    """

    if not self.built:
      self.build(x0.shape)

    if x is None:
      x = x0

    if x0.shape[-1] != x.shape[-1]:
      raise ValueError(
          "`x0` and `x` dimension mismatch! Got `x0` dimension {}, and x "
          "dimension {}. This case is not supported yet.".format(
              x0.shape[-1], x.shape[-1]))

    prod_output = tf.matmul(x, self._weight)
    return x0 * prod_output + self._bias + x

  def get_config(self):
    config = {
        "kernel_initializer":
            tf.keras.initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            tf.keras.initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            tf.keras.regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            tf.keras.regularizers.serialize(self._bias_regularizer),
    }
    base_config = super(CrossLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

