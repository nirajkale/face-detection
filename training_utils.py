import tensorflow as tf
import mlflow
import json
from os import path
import os
from object_detection.utils import label_map_util
from object_detection import eval_util
from object_detection.model_lib_v2 import prepare_eval_dict
from object_detection.protos import input_reader_pb2
from object_detection.builders.dataset_builder import read_dataset
import functools
from object_detection.builders import decoder_builder
from object_detection.utils import config_util
from object_detection.builders import image_resizer_builder
from object_detection.inputs import (get_reduce_to_frame_fn, \
    transform_input_data, \
    pad_input_data_to_static_shapes,\
    _get_features_dict, _get_labels_dict,\
    augment_input_data)
from object_detection.builders import preprocessor_builder

class MLFlowLogging(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            mlflow.log_metric(key, value, step=epoch)

def log_artifact(name, data, use_json=True):
    if use_json:
        fname = name+'.json'
        with open(fname, 'w') as f:
            f.write(json.dumps(data, indent=4))
    else:
        fname = name+'.txt'
        with open(fname, 'w') as f:
            f.write(data)
    mlflow.log_artifact(fname)
    os.remove(fname)

def get_or_create_mlflow_experiment(prefix, model_name):
    experiment_name = f'{prefix}-{model_name}'
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(experiment_name, )
        exp = mlflow.get_experiment_by_name(experiment_name)
    return exp

def dataset_map_fn(dataset, fn_to_map, batch_size=None, input_reader_config=None):
    if hasattr(dataset, 'map_with_legacy_function'):
        if batch_size:
            num_parallel_calls = batch_size * (input_reader_config.num_parallel_batches)
        else:
            num_parallel_calls = input_reader_config.num_parallel_map_calls
        dataset = dataset.map_with_legacy_function(fn_to_map, num_parallel_calls=num_parallel_calls)
    else:
        dataset = dataset.map(fn_to_map, tf.data.experimental.AUTOTUNE)
    return dataset

def read_dataset_custom(file_read_func, input_files, config, num_readers, enable_repeat= False, repeat_count = None):
    filenames = tf.compat.v1.gfile.Glob(input_files)
    print('Reading record datasets for input file: %s' % input_files)
    print('Number of filenames to read: %s' % len(filenames))
    if not filenames:
        raise RuntimeError('Did not find any input files matching the glob pattern ''{}'.format(input_files))
    if num_readers > len(filenames):
        num_readers = len(filenames)
        print(f'num_readers has been reduced to ${num_readers} to match input file shards.')
        # print('num_readers has been reduced to %d to match input file ''shards.' % num_readers)
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if config.shuffle:
        filename_dataset = filename_dataset.shuffle(config.filenames_shuffle_buffer_size)
    elif num_readers > 1:
        print('`shuffle` is false, but the input data stream is ''still slightly shuffled since `num_readers` > 1.')
    if enable_repeat:
        assert(repeat_count is not None)
        filename_dataset = filename_dataset.repeat(repeat_count or None)
    records_dataset = filename_dataset.apply(
        tf.data.experimental.parallel_interleave(
            file_read_func,
            cycle_length=num_readers,
            block_length=config.read_block_length,
            sloppy=config.shuffle))
    if config.shuffle:
        records_dataset = records_dataset.shuffle(config.shuffle_buffer_size)
    return records_dataset

def build_eval_dataset(model, model_config, eval_config, eval_input_config, batch_size):
    decoder = decoder_builder.build(eval_input_config)
    reduce_to_frame_fn = get_reduce_to_frame_fn(eval_input_config, False)
    if not isinstance(eval_input_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not of type ' 'input_reader_pb2.InputReader.')
    config = eval_input_config.tf_record_input_reader
    dataset = read_dataset_custom(
        functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
        config.input_path[:], \
        eval_input_config,\
        num_readers=2,\
        enable_repeat=False
    )
    dataset = dataset_map_fn(dataset, decoder.decode, batch_size, eval_input_config)
    if reduce_to_frame_fn:
        dataset = reduce_to_frame_fn(dataset, dataset_map_fn, batch_size, eval_input_config)
    model_preprocess_fn = model.preprocess

    def transform_and_pad_input_data_fn(tensor_dict):
        """Combines transform and pad operation."""
        num_classes = config_util.get_number_of_classes(model_config)

        image_resizer_config = config_util.get_image_resizer_config(model_config)
        image_resizer_fn = image_resizer_builder.build(image_resizer_config)
        keypoint_type_weight = eval_input_config.keypoint_type_weight or None

        transform_data_fn = functools.partial(
            transform_input_data, 
            model_preprocess_fn = model_preprocess_fn,
            image_resizer_fn = image_resizer_fn,
            num_classes = num_classes,
            data_augmentation_fn = None,
            retain_original_image = eval_config.retain_original_images,
            retain_original_image_additional_channels = eval_config.retain_original_image_additional_channels,
            keypoint_type_weight = keypoint_type_weight
        )
        tensor_dict = pad_input_data_to_static_shapes(
            tensor_dict=transform_data_fn(tensor_dict),
            max_num_boxes=eval_input_config.max_number_of_boxes,
            num_classes=config_util.get_number_of_classes(model_config),
            spatial_image_shape=config_util.get_spatial_image_size(
                image_resizer_config),
            max_num_context_features=config_util.get_max_num_context_features(
                model_config),
            context_feature_length=config_util.get_context_feature_length(
                model_config))
        include_source_id = eval_input_config.include_source_id
        return (_get_features_dict(tensor_dict, include_source_id),_get_labels_dict(tensor_dict))

    dataset = dataset_map_fn(dataset, transform_and_pad_input_data_fn, batch_size, eval_input_config)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def build_training_dataset(model, model_config, train_config, train_input_config, batch_size):
    num_classes = config_util.get_number_of_classes(model_config)
    model_preprocess_fn = model.preprocess

    decoder = decoder_builder.build(train_input_config)
    reduce_to_frame_fn = get_reduce_to_frame_fn(train_input_config, False)
    if not isinstance(train_input_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not of type ' 'input_reader_pb2.InputReader.')
    dataset = read_dataset_custom(
        functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
        train_input_config.tf_record_input_reader.input_path[:], \
        train_input_config,\
        num_readers=2,\
        enable_repeat=False
    )
    dataset = dataset_map_fn(dataset, decoder.decode, batch_size, train_input_config)
    if reduce_to_frame_fn:
        dataset = reduce_to_frame_fn(dataset, dataset_map_fn, batch_size, train_input_config)
    model_preprocess_fn = model.preprocess

    def transform_and_pad_input_data_fn(tensor_dict):
        """Combines transform and pad operation."""
        data_augmentation_options = [
            preprocessor_builder.build(step)
            for step in train_config.data_augmentation_options
        ]
        data_augmentation_fn = functools.partial(
            augment_input_data,
            data_augmentation_options=data_augmentation_options)

        image_resizer_config = config_util.get_image_resizer_config(model_config)
        image_resizer_fn = image_resizer_builder.build(image_resizer_config)
        keypoint_type_weight = train_input_config.keypoint_type_weight or None
        transform_data_fn = functools.partial(
            transform_input_data, model_preprocess_fn=model_preprocess_fn,
            image_resizer_fn=image_resizer_fn,
            num_classes=num_classes,
            data_augmentation_fn=data_augmentation_fn,
            merge_multiple_boxes=train_config.merge_multiple_label_boxes,
            retain_original_image=train_config.retain_original_images,
            use_multiclass_scores=train_config.use_multiclass_scores,
            use_bfloat16=train_config.use_bfloat16,
            keypoint_type_weight=keypoint_type_weight)

        tensor_dict = pad_input_data_to_static_shapes(
            tensor_dict=transform_data_fn(tensor_dict),
            max_num_boxes=train_input_config.max_number_of_boxes,
            num_classes=num_classes,
            spatial_image_shape=config_util.get_spatial_image_size(image_resizer_config),
            max_num_context_features=config_util.get_max_num_context_features(model_config),
            context_feature_length=config_util.get_context_feature_length(model_config))
        include_source_id = train_input_config.include_source_id
        return (_get_features_dict(tensor_dict, include_source_id),_get_labels_dict(tensor_dict))

    dataset = dataset_map_fn(dataset, transform_and_pad_input_data_fn, batch_size, train_input_config)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

