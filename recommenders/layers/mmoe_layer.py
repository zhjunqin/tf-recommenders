
import tensorflow.compat.v1 as tf

from tensorflow.keras.layers import Layer,InputSpec
from tensorflow.keras import initializers, regularizers, constraints, activations
from tensorflow.python.summary import summary
from tensorflow.python.ops import nn


class MMoELayer(Layer):
  """
  Multi-gate Mixture-of-Experts model.
  """

  def __init__(self,
               experts_network,
               num_experts,
               num_tasks,
               use_expert_bias=True,
               use_gate_bias=True,
               expert_activation='relu',
               gate_activation='softmax',
               expert_bias_initializer='zeros',
               gate_bias_initializer='zeros',
               expert_bias_regularizer=None,
               gate_bias_regularizer=None,
               expert_bias_constraint=None,
               gate_bias_constraint=None,
               expert_kernel_initializer='glorot_uniform',
               gate_kernel_initializer='glorot_uniform',
               expert_kernel_regularizer=None,
               gate_kernel_regularizer=None,
               expert_kernel_constraint=None,
               gate_kernel_constraint=None,
               activity_regularizer=None,
               **kwargs):
    """
    Method for instantiating MMoE layer.

    :param num_experts: Number of experts
    :param num_tasks: Number of tasks
    :param experts_network: hidden_layers in expert within last units
    :param use_expert_bias: Boolean to indicate the usage of bias in the expert weights
    :param use_gate_bias: Boolean to indicate the usage of bias in the gate weights
    :param expert_activation: Activation function of the expert weights
    :param gate_activation: Activation function of the gate weights
    :param expert_bias_initializer: Initializer for the expert bias
    :param gate_bias_initializer: Initializer for the gate bias
    :param expert_bias_regularizer: Regularizer for the expert bias
    :param gate_bias_regularizer: Regularizer for the gate bias
    :param expert_bias_constraint: Constraint for the expert bias
    :param gate_bias_constraint: Constraint for the gate bias
    :param expert_kernel_initializer: Initializer for the expert weights
    :param gate_kernel_initializer: Initializer for the gate weights
    :param expert_kernel_regularizer: Regularizer for the expert weights
    :param gate_kernel_regularizer: Regularizer for the gate weights
    :param expert_kernel_constraint: Constraint for the expert weights
    :param gate_kernel_constraint: Constraint for the gate weights
    :param activity_regularizer: Regularizer for the activity
    :param kwargs: Additional keyword arguments for the Layer class
    """
    assert experts_network is not None
    assert num_experts is not None and num_experts > 0
    assert num_tasks is not None and num_tasks > 0
    super(MMoELayer, self).__init__(**kwargs)

    # Hidden nodes parameter
    self.num_experts = num_experts
    self.num_tasks = num_tasks
    self.experts_network = experts_network

    # Weight parameter
    self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
    self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
    self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
    self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
    self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
    self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)

    # Activation parameter
    self.expert_activation = activations.get(expert_activation)
    self.gate_activation = activations.get(gate_activation)

    # Bias parameter
    self.expert_bias = None
    self.gate_bias = None
    self.use_expert_bias = use_expert_bias
    self.use_gate_bias = use_gate_bias
    self.expert_bias_initializer = initializers.get(expert_bias_initializer)
    self.gate_bias_initializer = initializers.get(gate_bias_initializer)
    self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
    self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
    self.expert_bias_constraint = constraints.get(expert_bias_constraint)
    self.gate_bias_constraint = constraints.get(gate_bias_constraint)

    # Activity parameter
    self.activity_regularizer = regularizers.get(activity_regularizer)

    # Keras parameter
    self.input_spec = InputSpec(min_ndim=2)
    self.supports_masking = True


  def build(self, input_shape):
    """
    Method for creating the layer weights.

    :param input_shape: Keras tensor (future input to layer)
                        or list/tuple of Keras tensors to reference
                        for weight shape computations
    """
    assert input_shape is not None and len(input_shape) >= 2

    tf.logging.info("mmoe layer input_shape: {}".format(input_shape))
    input_dimension = input_shape[-1]
    tf.logging.info("mmoe layer input_dimension: {}".format(input_dimension))
    tf.logging.info("expert networks: {}".format(self.experts_network))
    tf.logging.info("num_experts: {}".format(self.num_experts))

    # Initialize expert weights (number of input features * number of units per expert * number of experts)
    cur_input_dimension = input_dimension
    for index, num_node in enumerate(self.experts_network):
      if index > 0:
        cur_input_dimension = self.experts_network[index - 1]

      for expert_num in range(self.num_experts):
        name = 'expert_{}_hidden_{}'.format(expert_num, index)
        expert_layer = tf.keras.layers.Dense(name=name,
                                             units=num_node,
                                             activation=self.expert_activation,
                                             use_bias=self.use_expert_bias,
                                             kernel_initializer=self.expert_kernel_initializer,
                                             kernel_regularizer=self.expert_kernel_regularizer,
                                             bias_initializer=self.expert_bias_initializer,
                                             bias_regularizer=self.expert_bias_regularizer,
                                             kernel_constraint=self.expert_kernel_constraint,
                                             bias_constraint=self.expert_bias_constraint)
        expert_layer.build(cur_input_dimension)
        setattr(self, name, expert_layer)

    # Initialize gate weights (number of input features * number of experts * number of tasks)
    for i in range(self.num_tasks):
      name = 'gate_{}_hidden'.format(i)
      gate_layer = tf.keras.layers.Dense(name=name,
                                         units=self.num_experts,
                                         activation=self.gate_activation,
                                         use_bias=self.use_gate_bias,
                                         kernel_initializer=self.gate_kernel_initializer,
                                         kernel_regularizer=self.gate_kernel_regularizer,
                                         bias_initializer=self.gate_bias_initializer,
                                         bias_regularizer=self.gate_bias_regularizer,
                                         kernel_constraint=self.gate_kernel_constraint,
                                         bias_constraint=self.gate_bias_constraint)
      gate_layer.build(input_dimension)
      setattr(self, name, gate_layer)


    self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dimension})

    self.built = True

  def call(self, inputs):
    """
    Method for the forward function of the layer.

    :param inputs: Input tensor
    :return: A tensor
    """

    # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper
    expert_outputs = []
    for expert_num in range(self.num_experts):
      next_inputs = inputs
      for index, num_node in enumerate(self.experts_network):
        expert_layer = getattr(self, 'expert_{}_hidden_{}'.format(expert_num, index))
        next_inputs = expert_layer(next_inputs)
      expert_outputs.append(next_inputs)
    expert_outputs = tf.stack(expert_outputs, axis=0)

    # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper
    gate_outputs = []
    for i in range(self.num_tasks):
      gate_layer = getattr(self, 'gate_{}_hidden'.format(i))
      gate_output = gate_layer(inputs)
      gate_outputs.append(gate_output)

    # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
    final_outputs = []
    for gate_output in gate_outputs:
      output_sum = tf.einsum('ai,ian->an', gate_output, expert_outputs)
      final_outputs.append(output_sum)

    tf.logging.info("mmoe layer final_outputs :{}".format(final_outputs))

    return final_outputs

  def compute_output_shape(self, input_shape):
    """
    Method for computing the output shape of the MMoE layer.

    :param input_shape: Shape tuple (tuple of integers)
    :return: List of input shape tuple where the size of the list is equal to the number of tasks
    """
    assert input_shape is not None and len(input_shape) >= 2

    output_shape = list(input_shape)
    output_shape[-1] = self.experts_network[-1]
    output_shape = tuple(output_shape)
    shape = tf.TensorShape([output_shape for _ in range(self.num_tasks)])
    return shape

  def get_config(self):
    """
    Method for returning the configuration of the MMoE layer.

    :return: Config dictionary
    """
    cur_config = {
        'experts_network': self.experts_network,
        'num_experts': self.num_experts,
        'num_tasks': self.num_tasks,
        'use_expert_bias': self.use_expert_bias,
        'use_gate_bias': self.use_gate_bias,
        'expert_activation': activations.serialize(self.expert_activation),
        'gate_activation': activations.serialize(self.gate_activation),
        'expert_bias_initializer': initializers.serialize(self.expert_bias_initializer),
        'gate_bias_initializer': initializers.serialize(self.gate_bias_initializer),
        'expert_bias_regularizer': regularizers.serialize(self.expert_bias_regularizer),
        'gate_bias_regularizer': regularizers.serialize(self.gate_bias_regularizer),
        'expert_bias_constraint': constraints.serialize(self.expert_bias_constraint),
        'gate_bias_constraint': constraints.serialize(self.gate_bias_constraint),
        'expert_kernel_initializer': initializers.serialize(self.expert_kernel_initializer),
        'gate_kernel_initializer': initializers.serialize(self.gate_kernel_initializer),
        'expert_kernel_regularizer': regularizers.serialize(self.expert_kernel_regularizer),
        'gate_kernel_regularizer': regularizers.serialize(self.gate_kernel_regularizer),
        'expert_kernel_constraint': constraints.serialize(self.expert_kernel_constraint),
        'gate_kernel_constraint': constraints.serialize(self.gate_kernel_constraint),
        'activity_regularizer': regularizers.serialize(self.activity_regularizer)
        }
    config = super(MMoELayer, self).get_config()
    config.update(cur_config)
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)

