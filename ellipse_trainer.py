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
from collections import defaultdict
from object_detection.utils import visualization_utils as vutils
from object_detection.model_lib_v2 import _compute_losses_and_predictions_dicts
from tqdm import tqdm
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
        experiment_instance,
        run_name:str,
        epochs:int,
        n_training_samples:int,
        n_val_samples:int,
        pipeline_config_path,
        model_dir:str,
        save_final_config=False,
        **kwargs
    ):
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP['get_configs_from_pipeline_file']
    merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP['merge_external_params_with_configs']
    create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP['create_pipeline_proto_from_configs']

    configs = get_configs_from_pipeline_file(pipeline_config_path, config_override=None)
    configs = merge_external_params_with_configs(configs, None, kwargs_dict=kwargs)
    #load all the configs
    model_config = configs['model']
    train_config = configs['train_config']
    train_input_config = configs['train_input_config']
    eval_config = configs['eval_config']
    eval_input_configs = configs['eval_input_configs']
    eval_input_config = eval_input_configs[0]
    #done
    steps_per_epoch = math.ceil(n_training_samples/ train_config.batch_size)
    validation_steps = math.ceil(n_val_samples/ train_config.batch_size)
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
    dataset_adapter_train = inputs.train_input(
        train_config=train_config,
        train_input_config=train_input_config,
        model_config=model_config,
        model=detection_model,
        input_context=None
    )
    dataset_adapter_val = inputs.eval_input(
          eval_config=eval_config,
          eval_input_config=eval_input_config,
          model_config=model_config,
          model=detection_model
    )

    #input_context is None if training setup is not distributed
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
        total_loss = losses_dict['Loss/total_loss']
        gradients = tape.gradient(total_loss, detection_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, detection_model.trainable_variables))
        global_step.assign_add(1)
        return losses_dict

    @tf.function
    def compute_eval_dict(features, labels):
        """Compute the evaluation result on an image."""
        # For evaling on train data, it is necessary to check whether groundtruth
        # must be unpadded.
        boxes_shape = (labels[fields.InputDataFields.groundtruth_boxes].get_shape().as_list())
        unpad_groundtruth_tensors = (boxes_shape[1] is not None and eval_config.batch_size == 1)
        groundtruth_dict = labels
        labels = model_lib.unstack_batch(labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)
        losses_dict, prediction_dict = _compute_losses_and_predictions_dicts(detection_model, features, labels, train_config.add_regularization_loss)
        prediction_dict = detection_model.postprocess(prediction_dict, features[fields.InputDataFields.true_image_shape])
        eval_features = {
            fields.InputDataFields.image: features[fields.InputDataFields.image],
            fields.InputDataFields.original_image: features[fields.InputDataFields.original_image],
            fields.InputDataFields.original_image_spatial_shape: features[fields.InputDataFields.original_image_spatial_shape],
            fields.InputDataFields.true_image_shape: features[fields.InputDataFields.true_image_shape],
            inputs.HASH_KEY: features[inputs.HASH_KEY],
        }
        return losses_dict, prediction_dict, groundtruth_dict, eval_features
    
    #evaluation specific objects
    evaluator_options = eval_util.evaluator_options_from_eval_config(eval_config)
    class_agnostic_category_index = (label_map_util.create_class_agnostic_category_index())
    class_agnostic_evaluators = eval_util.get_evaluators(eval_config, list(class_agnostic_category_index.values()), evaluator_options)
    class_aware_evaluators = None
    if eval_input_config.label_map_path:
        class_aware_category_index = (label_map_util.create_category_index_from_labelmap(eval_input_config.label_map_path))
        class_aware_evaluators = eval_util.get_evaluators(eval_config,list(class_aware_category_index.values()),evaluator_options)
    #end
    # logged_step = global_step.value()
    data_iterator_train = iter(dataset_adapter_train)
    data_iterator_val = iter(dataset_adapter_val)
    last_step_time = time.time()
    with mlflow.start_run(experiment_id= experiment_instance.experiment_id, run_name=run_name):
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': train_config.batch_size,
            'steps_per_epoch': steps_per_epoch,
            'validation_steps': validation_steps
        })
        mlflow.log_artifact(model_dir)
        for epoch in range(epochs):
            tf.print(f'Epoch {epoch} started')
            batch_wise_loss_dict = defaultdict(list)
            for step in tqdm(range(steps_per_epoch), desc='Training Over Batches'):
                features, labels = next(data_iterator_train)
                losses_dict = train_step_fn(features, labels)
                for k,v in losses_dict.items():
                    batch_wise_loss_dict[k+'_train'].append(v)
                if step > 2:
                    break
            for step in tqdm(range(validation_steps), desc='Eval Over Batches'):
                features, labels = next(data_iterator_val)
                losses_dict, prediction_dict, groundtruth_dict, eval_features = compute_eval_dict(features, labels)
                eval_dict, class_agnostic = prepare_eval_dict(prediction_dict, groundtruth_dict, eval_features)
                evaluators = class_agnostic_evaluators if class_agnostic else class_aware_evaluators
                for evaluator in evaluators:
                    evaluator.add_eval_dict(eval_dict)
                for k,v in losses_dict.items():
                    batch_wise_loss_dict[k+'_val'].append(v)
            eval_metrics = {}
            for evaluator in evaluators:
                eval_metrics.update(evaluator.evaluate())
            logged_dict = {}
            logged_dict['learning_rate'] = float(learning_rate_fn())
            for k,v_list in batch_wise_loss_dict.items():
                logged_dict[k]= float(np.mean(v_list))
            for k, v in eval_metrics.items():
                logged_dict[str(k)] = float(v)
            mlflow.log_metrics(logged_dict)
    #save model
    tf.print(f'Saving model')
    manager.save()

if __name__ == '__main__':

    prefix = '640x640'
    model_name = 'ssd_resnet50'
    run_name = 'run-3'
    n_training_samples = 2306
    n_val_samples = 539
    experiment_instance = get_or_create_mlflow_experiment(prefix, model_name)

    training_subroutine(experiment_instance,\
        run_name= run_name,\
        epochs=1, n_training_samples=n_training_samples, \
        n_val_samples= n_val_samples,\
        pipeline_config_path=r'training/pretrained/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config',\
        model_dir= r'models/ssd_resnet50_v1',\
        save_final_config= True
    )