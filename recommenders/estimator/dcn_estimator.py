import os

import numpy as np
import tensorflow.compat.v1 as tf

from estimator import BaseEstimator
from input_fn import input_fn
from models import DCNModel


class DCNEstimator(BaseEstimator):

  def __init__(self, run_config, model_config, flags):

    super(DCNEstimator, self).__init__(
        model_config=model_config, run_config=run_config, flags=flags)

    self._networks = model_config.networks
    self._feature_columns = model_config.feature_columns

  def build_estimator(self):

    feature_config = self._model_config.feature_columns

    user_column = [feature_config[key] for key in self._networks['user_input']]
    item_column = [feature_config[key] for key in self._networks['item_input']]
    tf.logging.info('user_input column: {}'.format(user_column))
    tf.logging.info('item_input column: {}'.format(item_column))

    feature_columns = user_column + item_column
    self._feature_columns = feature_columns

    cross_layer_num = self._networks['cross_layer_num']
    dnn_hidden_units = self._networks['dnn_hidden_layers']
    label_name = self._networks['label_name']
    weight_column = None
    self._model= DCNModel(label_name=label_name,
                          user_column=user_column,
                          item_column=item_column,
                          cross_layer_num=cross_layer_num,
                          dnn_hidden_units=dnn_hidden_units,
                          learning_rate=self._flags.learning_rate)

    my_estimator = tf.estimator.Estimator(model_fn=self._model.model_fn(),
                                          config=self._run_config,
                                          model_dir=self._flags.output_dir)
    return my_estimator

