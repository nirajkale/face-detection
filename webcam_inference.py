import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from os import path
tf.get_logger().setLevel('ERROR') 

MODEL_DIR=r'/datadrive/experiments/face-detection/models/ssd_resnet50/run-batch-21-steps-2200-lrwsteps-200/exported'

PATH_TO_SAVED_MODEL = path.join(MODEL_DIR, 'saved_model')
PATH_TO_CFG = path.join(MODEL_DIR, 'pipeline.config')
PATH_TO_LABELS = r'training/label_map.pbtext'

print('loadong saved model')
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
print('model loaded')





