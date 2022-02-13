# -*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf


def input_fn(file_path, batch_size, num_epochs, total_work_num, task_index,
             config, is_eval=False, parse_function=None):

    def parse_record_batch(proto):
      parsed_features = tf.parse_example(proto, features=config.all_features)

      labels = dict()
      for key in config.label_features:
        labels[key] = parsed_features.pop(key)

      return parsed_features, labels

    if not parse_function:
      parse_function = parse_record_batch

    file_list = tf.train.match_filenames_once(file_path)
    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    if not is_eval:
      dataset = dataset.shard(total_work_num, task_index)
      dataset = dataset.repeat(num_epochs)
      tf.logging.info(
          "input shard total num: {}, index: {}".format(total_work_num, task_index))
      tf.logging.info("input train epochs: {}".format(num_epochs))

    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename, "GZIP"),
        cycle_length=5,
        num_parallel_calls=5)
    dataset = dataset.shuffle(batch_size * 2)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(parse_function, num_parallel_calls=5).prefetch(batch_size * 5)

    return dataset


def test_input_fn(file_pattern, config_dir):
  from parse_config import Config
  tf.disable_v2_behavior()

  config = Config(config_dir)

  file_list = file_pattern
  batch_size = 1
  num_epochs = 1
  total_work_num = 1
  task_index = 0
  features = config.all_features
  label_features = config.label_features

  dataset = input_fn(file_list, batch_size, num_epochs, total_work_num, task_index, config)
  data_iter = dataset.make_initializable_iterator()
  examples = data_iter.get_next()

  with tf.Session() as sess:
    sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
    sess.run(data_iter.initializer)
    print(sess.run(examples))


if __name__ == '__main__':

    file_pattern = "/path/to/criteo/tfrecord/train/part-*"
    config_dir = "config/criteo/wide_deep"
    test_input_fn(file_pattern, config_dir)
