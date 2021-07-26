import tensorflow as tf
import mlflow
import json
from os import path
import os

create_experiment_name = lambda prefix,exp_attempt_counter,model_name: '{0}-{1}-{2}'.format(prefix, exp_attempt_counter, model_name)

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

def get_new_mlflow_experiment(prefix, model_name):
    exp_attempt_counter = 1
    experiment_name = create_experiment_name(prefix, exp_attempt_counter, model_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    while exp is not None:
        exp_attempt_counter += 1
        experiment_name = create_experiment_name(prefix, exp_attempt_counter, model_name)
        exp = mlflow.get_experiment_by_name(experiment_name)
    # artifact_location = path.join( path.dirname(path.dirname(__file__)), 'mlruns')
    mlflow.create_experiment(experiment_name, )
    # flag_experiment_exists = False
    exp = mlflow.get_experiment_by_name(experiment_name)
    return exp