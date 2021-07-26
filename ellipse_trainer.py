from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mlflow
import math
import time

import numpy as np
import tensorflow as tf

from object_detection import eval_util
from object_detection import inputs
from object_detection import model_lib
from object_detection.builders import optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import train_pb2
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import ops
from object_detection.utils import variables_helper
from object_detection.utils import visualization_utils as vutils
from object_detection.model_lib_v2 import (load_fine_tune_checkpoint, _compute_losses_and_predictions_dicts)
if __package__:
    from .training_utils import *
else:
    from training_utils import *

MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP
NUM_STEPS_PER_ITERATION = 100


RESTORE_MAP_ERROR_TEMPLATE = (
    'Since we are restoring a v2 style checkpoint'
    ' restore_map was expected to return a (str -> Model) mapping,'
    ' but we received a ({} -> {}) mapping instead.'
)

def clearn_temp_dir(filepath):
    if tf.io.gfile.exists(filepath) and tf.io.gfile.isdir(filepath):
        tf.io.gfile.rmtree(filepath)

def training_subroutine(
        epochs,
        n_training_samples,
        pipeline_config_path,
        model_dir,
        save_final_config=False,
        **kwargs
    ):
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP['get_configs_from_pipeline_file']
    merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP['merge_external_params_with_configs']
    create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP['create_pipeline_proto_from_configs']

    configs = get_configs_from_pipeline_file(pipeline_config_path, config_override=None)
    configs = merge_external_params_with_configs(configs, None, kwargs_dict=kwargs)
    model_config = configs['model']
    train_config = configs['train_config']
    train_input_config = configs['train_input_config']
    steps_per_epoch = math.ceil(n_training_samples/ train_config.batch_size)
    mlflow.log_params({
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch
    })
    config_util.update_fine_tune_checkpoint_type(train_config)
    # Write the as-run pipeline config to disk.
    if save_final_config:
        tf.print('Saving pipeline config file to directory {}'.format(model_dir))
        pipeline_config_final = create_pipeline_proto_from_configs(configs)
        config_util.save_pipeline_config(pipeline_config_final, model_dir)

    # Build the model, optimizer, and training input
    detection_model = MODEL_BUILD_UTIL_MAP['detection_model_fn_base'](model_config=model_config, \
        is_training=True,
        add_summaries=True
    )

    def train_input(input_context):
        """Callable to create train input."""
        # Create the inputs.
        train_input = inputs.train_input(
            train_config=train_config,
            train_input_config=train_input_config,
            model_config=model_config,
            model=detection_model,
            input_context=input_context)
        train_input = train_input.repeat()
        return train_input

    # train_input = strategy.experimental_distribute_datasets_from_function(
    #     train_dataset_fn)
    global_step = tf.Variable(0, trainable=False, dtype=tf.compat.v2.dtypes.int64, name='global_step')
    optimizer, (learning_rate,) = optimizer_builder.build(train_config.optimizer, global_step=global_step)

    if callable(learning_rate):
        learning_rate_fn = learning_rate
    else:
        learning_rate_fn = lambda: learning_rate

    ckpt = tf.compat.v2.train.Checkpoint(step=global_step, model=detection_model, optimizer=optimizer)
    manager = tf.compat.v2.train.CheckpointManager(ckpt, model_dir, max_to_keep=1)
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    ckpt.restore(latest_checkpoint)

    @tf.function
    def train_step_fn(features, labels):
        labels = model_lib.unstack_batch(labels, unpad_groundtruth_tensors=train_config.unpad_groundtruth_tensors)
        with tf.GradientTape() as tape:
            losses_dict, _ = _compute_losses_and_predictions_dicts(detection_model, features, labels, train_config.add_regularization_loss)
        trainable_variables = detection_model.trainable_variables
        total_loss = losses_dict['Loss/total_loss']
        gradients = tape.gradient(total_loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        global_step.assign_add(1)
        return losses_dict
    # logged_step = global_step.value()
    last_step_time = time.time()
    data_iterator = iter(train_input)
    for _ in range(epochs):
        time_taken_list = []
        for _ in range(steps_per_epoch):
            features, labels = data_iterator.next()
            losses_dict = train_step_fn(features, labels)
            time_taken = time.time() - last_step_time
            time_taken_list.append(time_taken)
            last_step_time = time.time()
            # mlflow.log_metric('steps_per_sec', steps_per_sec, step=global_step)
        logged_dict = losses_dict.copy()
        logged_dict['learning_rate'] = learning_rate_fn()
        logged_dict['steps_per_sec'] = np.mean(time_taken_list)
        logged_dict = {k:float(v) for k,v in logged_dict.items()}
        mlflow.log_metrics(logged_dict)
    #save model
    manager.save()

if __name__ == '__main__':

    prefix = 'first-custom-run'
    model_name = 'ssd_resnet50'
    n_training_samples = 2306
    experiment_instance = get_new_mlflow_experiment(prefix, model_name)
    training_subroutine(epochs=5, steps_per_epoch=)