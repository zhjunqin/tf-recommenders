import numpy as np

import tensorflow as tf


def embedding_dim(num_buckets):
  """empirical embedding dim"""
  return int(np.power(2, np.ceil(np.log(num_buckets** 0.25)) + 3))



def build_feature_columns(config):

    feature_columns = config.feature_columns

    while True :
      has_not_ready_column = False

      for key,column in feature_columns.items():
        if 'column' in column:
          continue

        if 'depends' not in column:
          # No dependency
          column_func = getattr(tf.feature_column, column['type'])
          parameters = column.get('parameters', dict())
          column['column'] = column_func(**parameters)
        else:
          # Has dependency
          depends = column['depends']
          not_ready = list(filter(lambda key: 'column' not in feature_columns[key], depends))
          if not_ready:
            has_not_ready_column = True
            continue

          parameters = column['parameters']
          column_func = getattr(tf.feature_column, column['type'])

          if column['type'] == 'embedding_column':
            key = depends[0]
            categorical_column = feature_columns[key]['column']
            parameters['categorical_column'] = categorical_column
            if 'dimension' not in parameters:
                parameters['dimension'] = embedding_dim(categorical_column.num_buckets)

            column['column'] = column_func(**parameters)

          if column['type'] in ['indicator_column', 'weighted_categorical_column']:
            key = depends[0]
            parameters['categorical_column'] = feature_columns[key]['column']
            column['column'] = column_func(**parameters)

          if column['type'] == 'shared_embedding_columns':
            columns = list(map(lambda key: feature_columns[key]['column'], depends))
            parameters['categorical_columns'] = columns
            column['column'] = column_func(**parameters)

          if column['type'] == 'crossed_column':
            columns = list(map(lambda key: feature_columns[key]['column'], depends))
            parameters['keys'] = columns
            column['column'] = column_func(**parameters)

      if not has_not_ready_column:
        break

    return feature_columns


def test_feature_column(config):

  build_feature_columns(config)
  column_dict = dict()

  print("=== Feature column no depends ===")
  for key,item in config.feature_columns.items():
      if 'depends' not in item:
        print(key, item['column'])

  print("=== Feature column with depends ===")
  for key,item in config.feature_columns.items():
      if 'depends' in item:
        print(key, item['column'])


if __name__ == '__main__':

    from config import Config
    test_feature_column(Config)
