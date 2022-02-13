#!/bin/sh

set -e

# Set Envs
ESTIMATOR_MODEL=dcn
DATE_TIME="2021-03-03"
LEARNING_RATE=0.01
CONFIG_DIR=${PWD}/recommenders/config/criteo/dcn_local
TRAIN_DIR=/data/criteo/train
TEST_DIR=/data/criteo/test
CKPT_DIR=/data/criteo/ckpt
EXPORT_MODEL_DIR=/data/criteo/model

echo "TRAIN_DIR: ${TRAIN_DIR}"
echo "TEST_DIR: ${TEST_DIR}"
echo "CKPT_DIR: ${CKPT_DIR}"
echo "EXPORT_MODEL_DIR: ${EXPORT_MODEL_DIR}"

# Run TF
export PYTHONPATH=$PYTHONPATH:$PWD/recommenders
cd $PWD/recommenders

python tf.py --train_dir "${TRAIN_DIR}" \
             --test_dir "${TEST_DIR}" \
             --output_dir "${CKPT_DIR}" \
             --export_model_dir "${EXPORT_MODEL_DIR}" \
             --config_dir ${CONFIG_DIR} \
             --estimator_model ${ESTIMATOR_MODEL} \
             --date_time ${DATE_TIME} \
             --learning_rate ${LEARNING_RATE}
