import os
import numpy as np
import yaml

import tensorflow.compat.v1 as tf
from tensorflow.python.ops import math_ops


DATA_TYPES = {
  'int64': tf.int64,
  'int32': tf.int32,
  'string': tf.string,
  'float32': tf.float32,
}


class Config(object):

  def __init__(self, config_dir):
    self.parse_bucket_size(config_dir)
    self.parse_quantile_boundaries(config_dir)
    self.parse_settings(config_dir)
    self.parse_input_features(config_dir)
    self.parse_feature_coulmn(config_dir)
    self.parse_networks(config_dir)

  def parse_bucket_size(self, config_dir):
    """
    bucket size file with format 'key\tsize', eg:
    c1      1460
    c2      583
    c3      10131227
    """
    file_name = os.path.join(config_dir, "bucket_size.txt")
    self._bucket_size = dict()
    if not os.path.isfile(file_name):
      # file not exist
      return
    with open(file_name, 'r') as f:
      for line in f:
        line = line.strip().split('\t')
        key = line[0]
        value = int(line[1])
        self._bucket_size[key] = value

  def parse_quantile_boundaries(self, config_dir):
    """
    quantile boundaryise file with format 'key\t[v1, v2, ...]', eg:
    i1      [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    i2      [-3, -1, -1, 8, 32, 33, 48, 56]
    """
    file_name = os.path.join(config_dir, "quantile_boundaries.txt")
    self._quantile_boundaries = dict()
    if not os.path.isfile(file_name):
      # file not exist
      return
    with open(file_name, 'r') as f:
      for line in f:
        line = line.strip().split('\t')
        key = line[0]
        value = eval(line[1])
        self._quantile_boundaries[key] = value

  def parse_input_features(self, config_dir):
    file_name = os.path.join(config_dir, "input_features.yaml")
    features = self._load_yaml(file_name)

    label_features = dict()
    input_features = dict()
    for key,values in features.items():
      parameters = dict()
      length_type = values['length_type']
      if length_type == 'FixedLenFeature':
        parameters['dtype'] = DATA_TYPES[values['data_type']]
        parameters['shape'] = values['shape']
        parameters['default_value'] = values.get('default_value', None)
        feature = tf.io.FixedLenFeature(**parameters)
      elif length_type == 'VarLenFeature':
        parameters['dtype'] = DATA_TYPES[values['data_type']]
        feature = tf.io.VarLenFeature(**parameters)

      if values['input_type'] == 'label':
        label_features[key] = feature
      elif values['input_type'] == 'feature':
        input_features[key] = feature

    self.label_features = label_features
    tf.logging.info("label_features: {}".format(label_features))
    self.input_features = input_features
    tf.logging.info("input_features: {}".format(input_features))
    self.all_features = {**label_features, **input_features}

  def parse_feature_coulmn(self, config_dir):
    file_name = os.path.join(config_dir, "feature_column.yaml")
    columns = self._load_yaml(file_name)

    feature_columns = dict()
    while True :
      has_not_ready_column = False

      for key,column in columns.items():
        if key in feature_columns:
          continue

        depends = column.get('depends', [])
        not_ready = list(filter(lambda key: key not in feature_columns, depends))
        if not_ready:
          has_not_ready_column = True
          continue

        column_type = column['type']
        parameters = dict()
        if column_type == 'categorical_column_with_hash_bucket':
          parameters['key'] = column['key']
          parameters['hash_bucket_size'] = self.get_bucket_size(
              column['key'], column.get('hash_bucket_size'))
          parameters['dtype'] = DATA_TYPES[column['data_type']]

        if column_type == 'categorical_column_with_identity':
          parameters['key'] = column['key']
          parameters['num_buckets'] = self.get_bucket_size(
              column['key'], column.get('num_buckets'))
          parameters['default_value'] = column.get('default_value', None)

        if column_type == 'categorical_column_with_vocabulary_list':
          parameters['key'] = column['key']
          parameters['vocabulary_list'] = column['vocabulary_list']
          parameters['dtype'] = DATA_TYPES[column['data_type']]

        if column_type == 'numeric_column':
          parameters['key'] = column['key']
          parameters['shape'] = tuple(column.get('shape', [1,]))
          parameters['dtype'] = DATA_TYPES.get(column['data_type'], 'float32')
          parameters['default_value'] = column.get('default_value', None)
          normalizer_fn = column.get('normalizer_fn', None)
          if normalizer_fn:
            parameters['normalizer_fn'] = self.get_normalizer_fn(
                column['key'], column['data_type'], normalizer_fn)

        if column_type == 'embedding_column':
          categorical_column = feature_columns[depends[0]]
          parameters['categorical_column'] = categorical_column
          parameters['combiner'] = column.get('combiner', 'sqrtn')
          parameters['initializer'] = tf.uniform_unit_scaling_initializer()
          if 'dimension' not in column:
            parameters['dimension'] = self.get_embedding_dim(categorical_column.num_buckets)
          else:
            parameters['dimension'] = column['dimension']

        if column_type == 'indicator_column':
          categorical_column = feature_columns[depends[0]]
          parameters['categorical_column'] = categorical_column

        if column_type == 'shared_embedding_columns':
          column_list = list(map(lambda key: feature_columns[key], column['depends']))
          parameters['categorical_columns'] = column_list
          parameters['dimension'] = column['dimension']

        if column_type == 'crossed_column':
          column_list = list(map(lambda key: feature_columns[key], column['depends']))
          parameters['keys'] = column_list
          parameters['hash_bucket_size'] = column['hash_bucket_size']

        column_func = getattr(tf.feature_column, column_type)
        feature_columns[key] = column_func(**parameters)

      if not has_not_ready_column:
        break

    self.feature_columns = feature_columns
    tf.logging.info("feature_columns: {}".format(feature_columns))

  def parse_networks(self, config_dir):
    file_name = os.path.join(config_dir, "networks.yaml")
    self.networks = self._load_yaml(file_name)
    tf.logging.info("networks: {}".format(self.networks))

  def parse_settings(self, config_dir):
    file_name = os.path.join(config_dir, "settings.yaml")
    self.settings = self._load_yaml(file_name)
    tf.logging.info("settings: {}".format(self.settings))

  def _load_yaml(self, file_name):
    with open(file_name, 'r') as f:
      return yaml.safe_load(f)

  def get_bucket_size(self, key, num_buckets):
    if num_buckets is not None:
      # num_buckets setted in yaml
      return num_buckets
    if key not in self._bucket_size:
        raise Exception("Not found '{}' in bucket size file".format(key))
    bucket_size = self._bucket_size[key] + 1
    tf.logging.info("feature '{}' bucket_size: {}".format(key, bucket_size))
    return bucket_size

  def get_quantile_boundaries(self, key, dtype, boundaries):
    if boundaries is not None:
      # boundaries setted in yaml
      return boundaries
    if key not in self._quantile_boundaries:
      raise Exception("Not found '{}' in quantile boundarise file".format(key))
    if dtype == 'int64':
      boundaries = [int(i) for i in self._quantile_boundaries[key]]
    elif dtype == 'float32':
      boundaries = [float(i) for i in self._quantile_boundaries[key]]
    else:
      raise Exception("Wrong quantile boundaries data type: {}".format(dtype))
    tf.logging.info("feature '{}' boundaries: {}".format(key, boundaries))
    return boundaries

  def get_embedding_dim(self, num_buckets):
    """empirical embedding dim"""
    dim = int(np.power(2, np.ceil(np.log(num_buckets** 0.25)) + 1))
    return min(dim, 128)

  def get_normalizer_fn(self, key, dtype, normalizer_fn):
    if normalizer_fn['type'] == "LOG":
      def log(x):
        return tf.log(tf.cast(x, tf.float32) + tf.constant(1.0, tf.float32))

      return log

    if normalizer_fn['type'] == "QUANTILE":
      boundaries = normalizer_fn.get('boundaries', None)
      boundaries = self.get_quantile_boundaries(key, dtype, boundaries)

      def quantile(x):
        bucketize = math_ops._bucketize(x, boundaries=boundaries)
        return tf.math.divide(bucketize, len(boundaries) + 1)

      return quantile

    if normalizer_fn['type'] == "MAX_MIN":
      pass


if __name__ == '__main__':

  tf.logging.set_verbosity(tf.logging.INFO)
  file_dir = "config/criteo/dcn"
  config = Config(file_dir)
