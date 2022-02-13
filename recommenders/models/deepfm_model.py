import tensorflow.compat.v1 as tf

from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import multi_class_head
from tensorflow_estimator.python.estimator.head import multi_head
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head

from tensorflow.python.feature_column import feature_column

from layers import FMLayer


class DeepFMModel(object):
  """Deep FM
  NOTE: deep fm 模型的所有输入的 Embedding 大小必须保持一致
  """


  def __init__(self, label_name, user_embedding, user_dense,
               item_embedding, item_dense,
               dnn_hidden_units, learning_rate):
    self._label_name = label_name
    self._user_embedding = user_embedding
    self._user_dense = user_dense
    self._item_embedding = item_embedding
    self._item_dense = item_dense
    self._dnn_hidden_units = dnn_hidden_units
    self._learning_rate = learning_rate

  def build_layer(self):

    def dense_feature(column):
        return tf.keras.layers.DenseFeatures(column)

    self._user_embedding_input = [dense_feature(column) for column in self._user_embedding]
    self._user_dense_input = [dense_feature(column) for column in self._user_dense]
    self._item_embedding_input = [dense_feature(column) for column in self._item_embedding]
    self._item_dense_input = [dense_feature(column) for column in self._item_dense]

    # DNN part
    self._dnn_hidden_layers = []
    with tf.variable_scope("deepfm_dnn"):
      length = len(self._dnn_hidden_units)
      for layer_id, num_hidden_units in enumerate(self._dnn_hidden_units):
        layer_name = 'dnn_hidden_{}'.format(layer_id)
        hidden_layer = tf.keras.layers.Dense(
            units=num_hidden_units,
            activation='relu',
            name=layer_name)
        self._dnn_hidden_layers.append(hidden_layer)

      last_layer = tf.keras.layers.Dense(
          units=1, activation=None, name="dnn_hidden_{}".format(length))
      self._dnn_hidden_layers.append(last_layer)

    # FM part
    self._fm_layer = FMLayer()

  def model_fn(self):

    def _model_fn(features, labels, mode):
      self.build_layer()

      with tf.variable_scope("input_features"):
        user_embedding_input = [func(features) for func in self._user_embedding_input]
        user_dense_input = [func(features) for func in self._user_dense_input]
        item_embedding_input = [func(features) for func in self._item_embedding_input]
        item_dense_input = [func(features) for func in self._item_dense_input]

      # DNN
      with tf.variable_scope("dnn"):
        inputs = user_embedding_input + user_dense_input
        inputs = inputs + item_embedding_input + item_dense_input
        net = tf.concat(inputs, axis=1)
        for layer_id, hidden_layer in enumerate(self._dnn_hidden_layers):
          net = hidden_layer(net)
        dnn = net

      # FM
      with tf.variable_scope("fm"):
        embeddings = user_embedding_input + item_embedding_input
        # batch_size * field_size * embedding_size
        fm_input = tf.stack(embeddings, axis=1)
        fm = self._fm_layer(fm_input)

      output = tf.reduce_sum(tf.concat([fm, dnn], axis=1), axis=1, keepdims=True)

      tf.logging.info("=DeepFM_Model= output: {}".format(output))

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
