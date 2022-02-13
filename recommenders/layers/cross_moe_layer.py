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


from typing import Union, Text, Optional

import tensorflow.compat.v1 as tf

from tensorflow.keras.layers import Layer,InputSpec
from tensorflow.keras import initializers, regularizers, constraints, activations
from tensorflow.python.summary import summary
from tensorflow.python.ops import nn


class CrossMoELayer(tf.keras.layers.Layer):
  """Cross MoE Layer in Deep & Cross Network to learn explicit feature interactions.
    References:
        1. [R. Wang et al.](https://arxiv.org/pdf/2008.13535.pdf)
          See Eq. (1) for full-rank and Eq. (2) for low-rank version.
    Example:
        ```python
        # after embedding layer in a functional model:
        input = tf.keras.Input(shape=(None,), name='index', dtype=tf.int64)
        x0 = tf.keras.layers.Embedding(input_dim=32, output_dim=6)
        x1 = CrossMoELayer(num_experts=4, projection_dim=64)(x0, x0)
        x2 = CrossMoELayer(num_experts=4, projection_dim=64)(x0, x1)
        logits = tf.keras.layers.Dense(units=10)(x2)
        model = tf.keras.Model(input, logits)
        ```
    Args:
        num_experts: expert number.
        projection_dim: project dimension to reduce the computational cost.
          Default is `None` such that a full (`input_dim` by `input_dim`) matrix
          W is used. If enabled, a low-rank matrix W = U*V will be used, where U
          is of size `input_dim` by `projection_dim` and V is of size
          `projection_dim` by `input_dim`. `projection_dim` need to be smaller
          than `input_dim`/2 to improve the model efficiency. In practice, we've
          observed that `projection_dim` = d/4 consistently preserved the
          accuracy of a full-rank version.
        kernel_initializer: Initializer to use on the kernel matrix.
        bias_initializer: Initializer to use on the bias vector.
        kernel_regularizer: Regularizer to use on the kernel matrix.
        bias_regularizer: Regularizer to use on bias vector.
    Input shape: A tuple of 2 (batch_size, `input_dim`) dimensional inputs.
    Output shape: A single (batch_size, `input_dim`) dimensional output.
  """

  def __init__(
      self,
      num_experts: int,
      projection_dim: int,
      kernel_initializer: Union[
          Text, tf.keras.initializers.Initializer] = "truncated_normal",
      bias_initializer: Union[Text,
                              tf.keras.initializers.Initializer] = "zeros",
      kernel_regularizer: Union[Text, None,
                                tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Union[Text, None,
                              tf.keras.regularizers.Regularizer] = None,
      **kwargs):

    super(CrossMoELayer, self).__init__(**kwargs)

    self._num_experts = num_experts
    self._projection_dim = projection_dim
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._input_dim = None

    self._supports_masking = True

  def build(self, input_shape):
    last_dim = input_shape[-1]

    if self._projection_dim < 0 or self._projection_dim > last_dim // 2:
      raise ValueError(
          "`projection_dim` should be smaller than last_dim / 2 to improve "
          "the model efficiency, and should be positive. Got "
          "`projection_dim` {}, and last dimension of input {}".format(
              self._projection_dim, last_dim))

    self._expert_v = []
    self._expert_c = []
    self._expert_u = []
    self._gate = []
    for expert_index in range(self._num_experts):
      name = 'expert_v_{}'.format(expert_index)
      dense_v = tf.keras.layers.Dense(
          name=name,
          units=self._projection_dim,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          activation='tanh',  # refer to paper
          use_bias=False)
      self._expert_v.append(dense_v)

      name = 'expert_c_{}'.format(expert_index)
      dense_c = tf.keras.layers.Dense(
          name=name,
          units=self._projection_dim,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          activation='tanh',  # refer to paper
          use_bias=False)
      self._expert_c.append(dense_c)

      name = 'expert_u_{}'.format(expert_index)
      dense_u = tf.keras.layers.Dense(
          name=name,
          units=last_dim,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=None,
          use_bias=True)
      self._expert_u.append(dense_u)

    gate = tf.keras.layers.Dense(
        name="gate",
        units=self._num_experts,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation='softmax',  # refer to paper
        use_bias=False)
    self._gate = gate

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

    ei_list = []
    for index in range(self._num_experts):
      xl = x
      v_x = self._expert_v[index](xl)  # bs * projection_dim
      c_v = self._expert_c[index](v_x)  # bs * projection_dim
      u_c = self._expert_u[index](c_v)  # bs * input_dim
      ei = x0 * u_c  # bs * input_dim
      ei_list.append(ei)

    experts = tf.stack(ei_list, axis=1)  # bs * num_experts * input_dim
    gate = self._gate(x)  # bs * num_experts
    gate_experts = tf.einsum('ai,ain->an', gate, experts)

    output = gate_experts + x  # bs * input_dim
    return output

  def get_config(self):
    config = {
        "num_expert":
            self._num_experts,
        "projection_dim":
            self._projection_dim,
        "kernel_initializer":
            tf.keras.initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            tf.keras.initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            tf.keras.regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            tf.keras.regularizers.serialize(self._bias_regularizer),
    }
    base_config = super(CrossMoELayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

