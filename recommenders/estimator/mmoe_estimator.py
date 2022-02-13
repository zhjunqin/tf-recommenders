import os

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python.feature_column import feature_column

from estimator.base_estimator import BaseEstimator
from models.mmoe_model import MMoEModel
from input_fn import input_fn


class MMoEEstimator(BaseEstimator):

  def __init__(self, run_config, model_config, flags):
    super(MMoEEstimator, self).__init__(
        model_config=model_config, run_config=run_config, flags=flags)

    self._networks = model_config.networks
    self._feature_columns = model_config.feature_columns

  def build_estimator(self):

    feature_config = self._model_config.feature_columns

    user_input = [feature_config[key] for key in self._networks['user_input_layer']]
    item_input = [feature_config[key] for key in self._networks['item_input_layer']]
    tf.logging.info('user_input column: {}'.format(user_input))
    tf.logging.info('item_input column: {}'.format(item_input))

    feature_columns = []
    feature_columns.extend(user_input)
    feature_columns.extend(item_input)

    self._feature_columns = feature_columns

    hidden_units = self._networks['input_hidden_layers']
    tower_task_specs = self._networks['towers']
    num_experts = self._networks['num_experts']
    experts_network = self._networks['experts_layers']
    weight_column = None
    self._model= MMoEModel(tower_task_specs,
                           weight_column,
                           feature_columns,
                           num_experts,
                           hidden_units,
                           experts_network,
                           learning_rate=self._flags.learning_rate)

    my_estimator = tf.estimator.Estimator(model_fn=self._model.model_fn(),
                                          config=self._run_config,
                                          model_dir=self._flags.output_dir)
    return my_estimator

