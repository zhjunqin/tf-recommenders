import os

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_estimator.python.estimator.canned.dnn_linear_combined import DNNLinearCombinedClassifierV2

from tensorflow.python.feature_column import feature_column

from estimator import BaseEstimator
from input_fn import input_fn


class WideDeepEstimator(BaseEstimator):

  def __init__(self, run_config, model_config, flags):

    super(WideDeepEstimator, self).__init__(
        model_config=model_config, run_config=run_config, flags=flags)

    self._networks = model_config.networks
    self._feature_columns = model_config.feature_columns

  def build_estimator(self):

    feature_config = self._model_config.feature_columns

    wide_columns = [feature_config[key] for key in self._networks['wide_columns']]
    deep_columns = [feature_config[key] for key in self._networks['deep_columns']]
    tf.logging.info('wide_columns: {}'.format(wide_columns))
    tf.logging.info('deep_columns: {}'.format(deep_columns))

    feature_columns = wide_columns + deep_columns
    self._feature_columns = feature_columns

    dnn_hidden_units = self._networks['dnn_hidden_layers']

    estimator = DNNLinearCombinedClassifierV2(
        model_dir=self._flags.output_dir,
        linear_feature_columns=wide_columns,
        linear_optimizer=tf.keras.optimizers.Ftrl(
            learning_rate=0.01,
            l1_regularization_strength=0.001,
            l2_regularization_strength=0.001),
        dnn_feature_columns=deep_columns,
        dnn_optimizer=tf.keras.optimizers.Adagrad(
            learning_rate=0.01,
            initial_accumulator_value=0.1),
        dnn_hidden_units=dnn_hidden_units,
        config=self._run_config)

    return estimator


  def input_fn(self, file_path, batch_size, num_epochs,
               total_work_num, task_index, config, is_eval):

    def parse_record_batch(proto):
      parsed_features = tf.parse_example(proto, features=config.all_features)

      labels = dict()
      for key in config.label_features:
        labels[key] = parsed_features.pop(key)

      return parsed_features, labels[self._networks['label_name']]

    return lambda: input_fn(file_path=file_path,
                            batch_size=batch_size,
                            num_epochs=num_epochs,
                            total_work_num=total_work_num,
                            task_index=task_index,
                            config=config,
                            is_eval=is_eval,
                            parse_function=parse_record_batch)



