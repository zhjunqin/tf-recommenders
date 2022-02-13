import tensorflow.compat.v1 as tf

from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import multi_class_head
from tensorflow_estimator.python.estimator.head import multi_head
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head

from tensorflow.python.feature_column import feature_column

from layers import CrossMoELayer
from layers import CrossLayerV2


class DCNV2MoEModel(object):
  """DCN-Mixture-of-Expert Model
  """

  def __init__(self, label_name, user_column, item_column,
               num_experts, projection_dim,
               cross_layer_num, dnn_hidden_units,
               learning_rate):
    self._label_name = label_name
    self._user_column = user_column
    self._item_column = item_column
    self._num_experts = num_experts
    self._projection_dim = projection_dim
    self._cross_layer_num = cross_layer_num
    self._dnn_hidden_units = dnn_hidden_units
    self._learning_rate = learning_rate

  def build_layer(self):

    self._user_input = tf.keras.layers.DenseFeatures(self._user_column)
    self._item_input = tf.keras.layers.DenseFeatures(self._item_column)

    # DNN part
    self._dnn_hidden_layers = []
    with tf.variable_scope("dnn"):
      length = len(self._dnn_hidden_units)
      for layer_id, num_hidden_units in enumerate(self._dnn_hidden_units):
        layer_name = 'dnn_hidden_{}'.format(layer_id)
        hidden_layer = tf.keras.layers.Dense(
            units=num_hidden_units,
            activation='relu',
            name=layer_name)
        self._dnn_hidden_layers.append(hidden_layer)

    # Cross MoE Network part
    self._cross_layers = []
    with tf.variable_scope("cross_moe_network"):
      for layer_id in range(self._cross_layer_num):
        layer_name = 'cross_moe_{}'.format(layer_id)
        layer = CrossMoELayer(num_experts=self._num_experts,
                              projection_dim=self._projection_dim,
                              name=layer_name)
        self._cross_layers.append(layer)

    # Final output
    self._final_layer = tf.keras.layers.Dense(units=1,
                                              activation=None,
                                              name="dcnv2_moe_final_layer")

  def model_fn(self):

    def _model_fn(features, labels, mode):
      self.build_layer()

      with tf.variable_scope("input_features"):
        user_input = self._user_input(features)
        item_input = self._item_input(features)

      inputs = tf.concat([user_input, item_input], axis=1)
      tf.logging.info("=DCNV2_MoE_Model= input: {}".format(inputs))

      # DNN
      with tf.variable_scope("dnn"):
        net = inputs
        for layer_id, hidden_layer in enumerate(self._dnn_hidden_layers):
          net = hidden_layer(net)
        dnn = net

      # Cross Network
      with tf.variable_scope("cross_moe_network"):
        x0 = inputs
        x1 = inputs
        for layer_id, cross_layer in enumerate(self._cross_layers):
          x1 = cross_layer(x0, x1)
        cross = x1

      output = self._final_layer(tf.concat([cross, dnn], axis=1))

      tf.logging.info("=DCNV2_MoE_Model= output: {}".format(output))

      if mode == tf.estimator.ModeKeys.PREDICT:
        clk = tf.sigmoid(output, name="clk")

        predictions = {"clk": clk}
        key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        export_outputs = {key: tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

      logits = dict()
      logits['clk'] = output
      optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate)
      head = binary_class_head.BinaryClassHead(name='clk')
      return head.create_estimator_spec(
          features=features,
          mode=mode,
          labels=labels[self._label_name],
          optimizer=optimizer,
          logits=logits['clk'])

    return _model_fn

