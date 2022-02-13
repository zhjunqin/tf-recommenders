import os

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python.feature_column import feature_column

from estimator.base_estimator import BaseEstimator
from models.deepfm_model import DeepFMModel
from input_fn import input_fn


class DeepFMEstimator(BaseEstimator):

  def __init__(self, run_config, model_config, flags):
    super(DeepFMEstimator, self).__init__(
        model_config=model_config, run_config=run_config, flags=flags)

    self._networks = model_config.networks
    self._feature_columns = model_config.feature_columns

  def build_estimator(self):

    feature_config = self._model_config.feature_columns

    user_embedding = [feature_config[key] for key in self._networks['user_embedding']]
    user_dense = [feature_config[key] for key in self._networks['user_dense']]
    item_embedding = [feature_config[key] for key in self._networks['item_embedding']]
    item_dense = [feature_config[key] for key in self._networks['item_dense']]
    tf.logging.info('user_input column: {}'.format(user_embedding))
    tf.logging.info('item_input column: {}'.format(user_dense))
    tf.logging.info('user_input column: {}'.format(item_embedding))
    tf.logging.info('item_input column: {}'.format(item_dense))

    feature_columns = user_embedding + user_dense + item_embedding + item_dense
    self._feature_columns = feature_columns

    dnn_hidden_units = self._networks['dnn_hidden_layers']
    label_name = self._networks['label_name']
    weight_column = None
    self._model= DeepFMModel(label_name=label_name,
                             user_embedding=user_embedding,
                             user_dense=user_dense,
                             item_embedding=item_embedding,
                             item_dense=item_dense,
                             dnn_hidden_units=dnn_hidden_units,
                             learning_rate=self._flags.learning_rate)

    my_estimator = tf.estimator.Estimator(model_fn=self._model.model_fn(),
                                          config=self._run_config,
                                          model_dir=self._flags.output_dir)
    return my_estimator

