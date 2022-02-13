# -*- coding:utf-8 -*-

import argparse
import sys

import grpc
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.example import feature_pb2, example_pb2

from parse_config import Config


FLAGS = None


def build_request(request, config, batch_size):
  feature_dict = dict()

  for key, item in config.input_features.items():
    dtype = item.dtype
    if isinstance(item, tf.io.FixedLenFeature):
      if dtype == tf.int64:
        feature_value = np.random.randint(0, 10, (batch_size, 1))
      if dtype == tf.float32:
        feature_value = np.random.randn(batch_size, 1)
      if dtype == tf.string:
        feature_value = np.random.randint(0, 10, (batch_size, 1)).astype(str)
    elif isinstance(item, tf.io.VarLenFeature):
      if dtype == tf.int64:
        feature_value = np.random.randint(0, 10, (batch_size, 2))

    tensor_proto = tf.make_tensor_proto(feature_value, dtype=dtype)
    request.inputs[key].CopyFrom(tensor_proto)


def main(_):
  config = Config(FLAGS.config_dir)
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model_name
  request.model_spec.signature_name = FLAGS.signature_name
  build_request(request, config, FLAGS.batch_size)

  result = stub.Predict(request, 10.0)
  print(result)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--server',
                        type=str,
                        required=True,
                        help='PredictionService host:port')
    parser.add_argument('--config_dir',
                        type=str,
                        required=True,
                        help='Dir for config yaml')
    parser.add_argument('--batch_size',
                        type=int,
                        required=False,
                        help='batch size',
                        default=2)
    parser.add_argument('--model_name',
                        type=str,
                        help='model name',
                        required=True)
    parser.add_argument('--signature_name',
                        type=str,
                        help='signature name',
                        required=False,
                        default='serving_default')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


