import tensorflow as tf
import mlflow
import json
from os import path
import os
from object_detection.utils import label_map_util
from object_detection import eval_util
from object_detection.model_lib_v2 import prepare_eval_dict

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