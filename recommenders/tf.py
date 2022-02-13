# -*- coding:utf-8 -*-

import argparse
import datetime
import json
import logging
import math
import os
import shutil
import sys
import time

import tensorflow.compat.v1 as tf

import estimator
from parse_config import Config
from util import std_flush, head

tf.disable_v2_behavior()
FLAGS = None


ESTIMATOR_MODEL = {
    'mmoe': estimator.MMoEEstimator,
    'deepfm': estimator.DeepFMEstimator,
    'dcn': estimator.DCNEstimator,
    'dcnv2': estimator.DCNV2Estimator,
    'dcnv2_moe': estimator.DCNV2MoEEstimator,
    'wide_deep': estimator.WideDeepEstimator,
}


def main(_):

    run_config = tf.estimator.RunConfig(
        keep_checkpoint_max=3,
        save_checkpoints_steps=55000,
        save_summary_steps=500,
        log_step_count_steps=500,
    ).replace(
        session_config=tf.ConfigProto(device_filters=['/job:ps'],
                                      intra_op_parallelism_threads=10,
                                      inter_op_parallelism_threads=10))

    parameters = {
        "model_config": Config(FLAGS.config_dir),
        "run_config": run_config,
        "flags": FLAGS}
    model = ESTIMATOR_MODEL[FLAGS.estimator_model](**parameters)

    estimator = model.build_estimator()

    train_hooks = []
    if run_config.cluster_spec:
        # add training hooks if any
        pass

    if run_config.task_type in ["chief", "master"]:
        # Do something only in chief or master
        pass

    train_spec = tf.estimator.TrainSpec(input_fn=model.train_input_fn(),
                                        max_steps=None,
                                        hooks=train_hooks)

    eval_spec = tf.estimator.EvalSpec(input_fn=model.eval_input_fn(),
                                      steps=10000,
                                      throttle_secs=300)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if run_config.task_type in ["chief", "master"] or not run_config.cluster_spec:
        # eval all data only in chief or master
        metric = estimator.evaluate(input_fn=model.eval_input_fn(),
                                    steps=None)
        tf.logging.info("final metric: {}".format(metric))

        # generate warmup request and export saved model
        warmup_requests_path = model.generate_warmup_requests(
            save_dir="/tmp/", serving_model_name=FLAGS.estimator_model)
        assets_extra = {'tf_serving_warmup_requests': warmup_requests_path}
        estimator.export_saved_model(
            export_dir_base=FLAGS.export_model_dir,
            serving_input_receiver_fn=model.raw_serving_input_receiver_tensor_fn(),
            assets_extra=assets_extra)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir',
                        type=str,
                        required=True,
                        help='Directory for training input data')
    parser.add_argument('--test_dir',
                        type=str,
                        required=True,
                        help='Directory for eval input data')
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        help='Directory for storing output')
    parser.add_argument('--export_model_dir',
                        type=str,
                        required=True,
                        help='Directory for storing export model')
    parser.add_argument('--config_dir',
                        type=str,
                        required=True,
                        help='Directory for storing config yaml files')
    parser.add_argument('--hook',
                        action='store_true',
                        default=False,
                        help='profile hook')
    parser.add_argument('--hook_dir',
                        type=str,
                        default='./hook',
                        help='Directory for storing profile info')
    parser.add_argument('--date_time',
                        type=str,
                        required=True,
                        help='Date time')
    parser.add_argument('--estimator_model',
                        type=str,
                        required=True,
                        help='model type')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='Initial learning rate')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
