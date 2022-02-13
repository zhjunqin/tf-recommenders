import abc
import os

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python.feature_column import feature_column
from tensorflow.python.saved_model import signature_constants

from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import multi_head
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head

from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from input_fn import input_fn


class BaseEstimator(object):

  def __init__(self, model_config, run_config, flags):
    self._run_config = run_config
    self._model_config = model_config
    self._flags = flags

  @abc.abstractmethod
  def build_estimator(self):
    raise NotImplementedError('Must be implemented in subclasses.')

  def input_fn(self, file_path, batch_size, num_epochs,
               total_work_num, task_index, config, is_eval):

    return lambda: input_fn(file_path=file_path,
                            batch_size=batch_size,
                            num_epochs=num_epochs,
                            total_work_num=total_work_num,
                            task_index=task_index,
                            config=config,
                            is_eval=is_eval,
                            )

  def train_input_fn(self):
    config = self._model_config
    setting = self._model_config.settings
    train_path = "{}/part-*".format(self._flags.train_dir)
    total_workers = self._run_config.num_worker_replicas
    worker_index = self._run_config.global_id_in_cluster
    tf.logging.info(
        "train_input_fn path {}: {}/{}".format(train_path, worker_index, total_workers))

    return self.input_fn(file_path=train_path,
                         batch_size=setting['batch_size'],
                         num_epochs=setting['train_epochs'],
                         total_work_num=total_workers,
                         task_index=worker_index,
                         config=config,
                         is_eval=False)

  def eval_input_fn(self):
    config = self._model_config
    setting = self._model_config.settings
    eval_path = "{}/part-*".format(self._flags.test_dir)
    total_workers = 1
    worker_index = 1
    tf.logging.info(
        "eval_input_fn path {}: {}/{}".format(eval_path, worker_index, total_workers))

    return self.input_fn(file_path=eval_path,
                         batch_size=setting['batch_size'],
                         num_epochs=1,
                         config=config,
                         total_work_num=1,
                         task_index=0,
                         is_eval=True)

  def raw_serving_input_receiver_tensor_fn(self):
    config = self._model_config
    feature_spec = {}

    for key, item in config.input_features.items():
      dtype = item.dtype
      feature_spec[key] = tf.placeholder(dtype=dtype, shape=[None, None])

    tf.logging.info("feature_spec: {}".format(feature_spec))

    return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

  def parsing_serving_input_receiver_fn(self):
    feature_spec = tf.feature_column.make_parse_example_spec(self._feature_columns)
    tf.logging.info("feature_spec: {}".format(feature_spec))
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

  def one_parsing_request(self, serving_model_name, signature_name, batch_size=10):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = serving_model_name
    request.model_spec.signature_name = signature_name
    config = self._model_config

    feature_dict = dict()
    for key, item in config.input_features.items():
      dtype = item.dtype
      if isinstance(item, tf.FixedLenFeature):
        if dtype == tf.int64:
          feature_value = np.random.randint(0, 10, (batch_size, 1))
        if dtype == tf.float32:
          feature_value = np.random.randn(batch_size, 1)
        if dtype == tf.string:
          feature_value = np.random.randint(0, 10, (batch_size, 1)).astype(str)

      elif isinstance(item, tf.VarLenFeature):
        if dtype == tf.int64:
          feature_value = np.random.randint(0, 10, (batch_size, 2))

      tensor_proto = tf.make_tensor_proto(feature_value, dtype=dtype)
      request.inputs[key].CopyFrom(tensor_proto)

    return request

  def generate_warmup_requests(
      self, save_dir, serving_model_name,
      signature_name=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
      num=100):
    warmup_file_path = os.path.join(save_dir, "tf_serving_warmup_requests")

    with tf.io.TFRecordWriter(warmup_file_path) as writer:
      for i in range(num):
        request = self.one_parsing_request(serving_model_name, signature_name)
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())

    return warmup_file_path

