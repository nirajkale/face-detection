download original images from: http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz

unzip:
tar -xvzf originalPics.tar.gz images

unzip pretrained model

tar -xvzf training/pretrained/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

total training sample: 2306/40 => 58 steps per epoch | 25000/58=> 431 epochs
total training sample: 2306/40 => 58 steps per epoch | 14 epochs=> 2900 steps
total eval samples: 539

training command => 

nohup python trainer.py --model_dir=models/my_ssd_resnet50_v1_fpn/steps-14_batch-28 \
 --pipeline_config_path=training/pretrained/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config &

tensorboard:

tensorboard --logdir=/datadrive/experiments/face-detection/models/ssd_resnet50_v1_fpn --port=9095

buider MAP:
ssd : <function _build_ssd_model at 0x7f3925353730>
faster_rcnn : <function _build_faster_rcnn_model at 0x7f39253538c8>
experimental_model : <function _build_experimental_model at 0x7f3925353950>
center_net : <function _build_center_net_model at 0x7f3925353d90>

model-builder-> def _build_ssd_model(ssd_config, is_training, add_summaries):
