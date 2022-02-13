import tensorflow.compat.v1 as tf

from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import multi_class_head
from tensorflow_estimator.python.estimator.head import multi_head
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head

from tensorflow.python.feature_column import feature_column

from layers import MMoELayer


class MMoEModel(object):

  def __init__(self, tower_task_specs, weight_column, feature_columns,
               num_experts, input_hidden_units, experts_network,
               learning_rate):
    self._tower_task_specs = tower_task_specs
    self._weight_column = weight_column
    self._feature_columns = feature_columns
    self._num_experts = num_experts
    self._input_hidden_units = input_hidden_units
    self._experts_network = experts_network
    self._learning_rate = learning_rate
    tf.logging.info("=MMoEModel= tower_task_specs: {}".format(tower_task_specs))


  def build_layer(self):
    # Input layer
    self._input_layer = feature_column.InputLayer(
        feature_columns=self._feature_columns, name='input_layer')

    # Input hidden
    self._input_hidden_layers = []
    for layer_id, num_hidden_units in enumerate(self._input_hidden_units):
      layer_name = 'input_hidden_{}'.format(layer_id)
      with tf.variable_scope(layer_name) as hidden_layer_scope:
        hidden_layer = tf.keras.layers.Dense(
            units=num_hidden_units,
            activation='relu',
            name=layer_name)
      self._input_hidden_layers.append(hidden_layer)

    # MMoE
    self._mmoe_layer = MMoELayer(
        experts_network=self._experts_network,
        num_experts=self._num_experts,
        num_tasks=len(self._tower_task_specs),
        name="mmoe_layer")

    # Tower
    self._tower = dict()
    for index, tower_task_spec in enumerate(self._tower_task_specs):
      for inner_index, num_nodes in enumerate(tower_task_spec['hidden_layers']):
        layer_name = 'tower_{}_hidden_{}'.format(index, inner_index)
        self._tower[layer_name] = tf.keras.layers.Dense(
            num_nodes, activation='relu', name=layer_name)

      self._tower['tower_{}'.format(index)] = tf.keras.layers.Dense(
          units=1, name='tower_{}'.format(index), activation=None)


  def model_fn(self):

    def _model_fn(features, labels, mode):
      self.build_layer()

      net = self._input_layer(features)
      for i in range(len(self._input_hidden_layers)):
        net = self._input_hidden_layers[i](net)

      output_layers = []
      mmoe_output = self._mmoe_layer(net)
      for index, task_layer in enumerate(mmoe_output):
        tower_layer = task_layer
        for inner_index in range(len(self._tower_task_specs[index]['hidden_layers'])):
          tower_layer = self._tower['tower_{}_hidden_{}'.format(index, inner_index)](tower_layer)

        output_layer = self._tower['tower_{}'.format(index)](tower_layer)
        output_layers.append(output_layer)

      # Combine logits and build full model.
      logits = dict()
      for index, tower_task_spec in enumerate(self._tower_task_specs):
        output_weight = tower_task_spec['output_weight']
        logits[tower_task_spec['target']] = output_layers[index] * output_weight

      tf.logging.info("=MMoEModel= logits: {}".format(logits))

      if mode == tf.estimator.ModeKeys.PREDICT:
        clk = tf.sigmoid(logits['clk'], name="clk")
        price = logits['price']

        predictions = {"price": price,
                       "clk": clk}
        key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        export_outputs = {key: tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

      optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate)

      head0 = regression_head.RegressionHead(name='price')
      head1 = binary_class_head.BinaryClassHead(name='clk')
      head = multi_head.MultiHead([head0, head1])

      return head.create_estimator_spec(
          features=features,
          mode=mode,
          labels=labels,
          optimizer=optimizer,
          logits=logits)

    return _model_fn
